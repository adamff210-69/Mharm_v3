"""Prompt construction, token-range mapping, and v3 §2.0 token-range validation.

This module is the single most likely silent-failure point of the pipeline
(v3 §2.0), so it is deliberately defensive:

* character spans of the passage / query / injection are located by literal
  substring search inside the rendered chat template (not by counting tokens
  of each part separately), then mapped to token indices with the fast
  tokenizer's offset mapping;
* ``validate_token_ranges`` re-decodes the token ranges and checks that the
  decoded text actually contains the expected substring for ~10-20 samples
  BEFORE any calibration runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

DEFAULT_SYSTEM = ("You are a helpful assistant. Use the provided passage to "
                  "answer the user's question accurately.")
PLAIN_TEMPLATE = "Context:\n{passage}\n\nQuestion: {query}\nAnswer:"


@dataclass
class Encoding:
    sample_id: str
    input_ids: torch.Tensor          # (T,)
    attention_mask: torch.Tensor     # (T,) all-ones (batch=1, no padding)
    passage_range: tuple             # (start, end) token indices (end exclusive)
    query_range: tuple
    inj_range: tuple | None          # injection span tokens (None for clean)
    n_tokens: int
    text: str = field(repr=False, default="")
    valid: bool = True
    note: str = ""


def _render_prompt(tokenizer, passage: str, query: str) -> str:
    """Render the chat template if the tokenizer supports one, else plain."""
    try:
        msgs = [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": f"Passage:\n{passage}\n\nQuery:\n{query}"},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{PLAIN_TEMPLATE.format(passage=passage, query=query)}"


def _span_to_tokens(offsets, char_span):
    """Map a (char_start, char_end) span to token indices via offsets.

    A token is 'inside' the span if it overlaps it; this is inclusive of the
    edge tokens (acceptable: the ratio is a mass fraction, and edge tokens are
    shared consistently between clean and injected samples).
    """
    cs, ce = char_span
    tok_s, tok_e = None, None
    for i, (s, e) in enumerate(offsets):
        if e <= cs or s >= ce:
            continue
        tok_s = i if tok_s is None else tok_s
        tok_e = i + 1
    return (tok_s, tok_e) if tok_s is not None else None


def encode_sample(tokenizer, sample: dict, max_seq_len: int,
                  tail_len: int = 48) -> Encoding:
    """Tokenize one sample and locate passage / query / injection ranges."""
    sid = sample["id"]
    passage, query = sample["passage"], sample["query"]
    text = _render_prompt(tokenizer, passage, query)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    ids = enc["input_ids"]

    def find_span(needle: str, start: int = 0):
        i = text.find(needle, start)
        return (i, i + len(needle)) if i != -1 else None

    p_span = find_span(passage)
    q_span = find_span(query, p_span[1] if p_span else 0)
    inj_span = None
    if sample.get("injection"):
        # injection offsets are passage-relative (from the dataset builder);
        # may come back as a numpy array after a parquet round-trip
        off = sample.get("injection_offset")
        off = None if off is None else list(off)
        if p_span and off is not None and off[0] is not None:
            inj_span = (p_span[0] + int(off[0]), p_span[0] + int(off[1]))
        else:
            inj_span = find_span(sample["injection"])

    valid, note = True, ""
    if p_span is None:
        valid, note = False, "passage substring not found in rendered prompt"
    elif q_span is None:
        valid, note = False, "query substring not found in rendered prompt"
    elif sample.get("injection") and inj_span is None:
        valid, note = False, "injection substring not found in rendered prompt"

    p_tok = _span_to_tokens(offsets, p_span) if p_span else None
    q_tok = _span_to_tokens(offsets, q_span) if q_span else None
    i_tok = _span_to_tokens(offsets, inj_span) if inj_span else None
    if valid and (p_tok is None or q_tok is None):
        valid, note = False, "token span mapping failed"

    # Clean samples: pseudo-injection region = last `tail_len` tokens of the
    # passage (documented design assumption, see README "Design decisions").
    if valid and i_tok is None and p_tok is not None:
        ps, pe = p_tok
        i_tok = (max(ps, pe - tail_len), pe)

    if len(ids) > max_seq_len:
        if valid and q_tok is not None and q_tok[1] > max_seq_len:
            valid, note = False, "query range truncated beyond max_seq_len"
        ids = ids[:max_seq_len]

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    return Encoding(
        sample_id=sid,
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        passage_range=p_tok or (0, 0),
        query_range=q_tok or (0, 0),
        inj_range=i_tok,
        n_tokens=ids.shape[1],
        text=text,
        valid=valid,
        note=note,
    )


# ---------------------------------------------------------------------------
# v3 §2.0 — validation
# ---------------------------------------------------------------------------

def _decode_range(tokenizer, enc: Encoding, rng: tuple) -> str:
    """Decode a token range. MUST use tokenizer.decode (not
    convert_ids_to_tokens + join): for BPE tokenizers (GPT-2, Llama) a join of
    raw sub-tokens leaves '##' artifacts and splits words, which makes the
    word-containment checks below fail even for correctly-aligned ranges."""
    ids = enc.input_ids[0][rng[0]:rng[1]].tolist()
    return tokenizer.decode(ids, skip_special_tokens=True)


def validate_token_ranges(tokenizer, samples: list[dict], max_seq_len: int,
                          tail_len: int) -> dict:
    """Encode ~N samples and verify decoded ranges contain the expected text.

    Returns a report dict; writes are done by the calling script.
    """
    results = []
    for s in samples:
        enc = encode_sample(tokenizer, s, max_seq_len, tail_len)
        rec = {"id": s["id"], "attack_type": s["attack_type"],
               "goal": s["goal"], "valid": enc.valid, "note": enc.note,
               "n_tokens": enc.n_tokens,
               "passage_range": list(enc.passage_range),
               "query_range": list(enc.query_range),
               "inj_range": list(enc.inj_range) if enc.inj_range else None,
               "decoded_passage_head": "", "decoded_passage_tail": "",
               "decoded_query_head": ""}
        if enc.valid:
            dp = _decode_range(tokenizer, enc, enc.passage_range)
            dq = _decode_range(tokenizer, enc, enc.query_range)
            rec["decoded_passage_head"] = dp[:80]
            rec["decoded_passage_tail"] = dp[-80:]
            rec["decoded_query_head"] = dq[:80]
            norm = lambda t: re.sub(r"\s+", " ", t).lower()
            # The decoded range must contain the bulk of the source text.
            src_p = norm(s["passage"])
            ok_p = (src_p[:60] in norm(dp) or src_p[-60:] in norm(dp)
                    or _containment(src_p, norm(dp)) > 0.8)
            ok_q = (norm(s["query"])[:40] in norm(dq)
                    or _containment(norm(s["query"]), norm(dq)) > 0.8)
            if enc.inj_range and s.get("injection"):
                di = _decode_range(tokenizer, enc, enc.inj_range)
                rec["decoded_inj_head"] = di[:80]
                ok_i = _containment(norm(s["injection"]), norm(di)) > 0.8
            else:
                ok_i = True
            rec["checks"] = {"passage": bool(ok_p), "query": bool(ok_q),
                             "injection": bool(ok_i)}
            rec["all_ok"] = bool(ok_p and ok_q and ok_i)
        else:
            rec["all_ok"] = False
        results.append(rec)
    n_ok = sum(1 for r in results if r["all_ok"])
    return {"n_samples": len(results), "n_ok": n_ok,
            "passed": n_ok == len(results) and len(results) > 0,
            "results": results}


def _containment(needle: str, haystack: str) -> float:
    """Fraction of needle's words that appear (in order, loosely) in haystack."""
    nw = needle.split()
    if not nw:
        return 1.0
    hits = sum(1 for w in nw if w in haystack)
    return hits / len(nw)
