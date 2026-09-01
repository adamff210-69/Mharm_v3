"""Dataset construction: clean MS-MARCO query-passage pairs + injected pairs
(4 attack types x 5 goals x 50) with explicit injection char offsets, and a
60/20/20 split stratified by (attack_type, goal) plus clean source.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from config import ATTACK_TYPES, ATTACK_WRAPPERS, GOALS

_STOP = set("the a an of and to in for is are was were on at by with from as or that this it be".split())


def _seed_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _truncate_chars(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:.") + "."


def _topic_keyword(passage: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z]{4,}", passage.lower()) if w not in _STOP]
    if not words:
        return "subject"
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return max(freq, key=freq.get)


# ---------------------------------------------------------------------------
# Clean pairs
# ---------------------------------------------------------------------------

def load_clean_pairs(cfg) -> pd.DataFrame:
    """Return df with columns [passage, query, source_id]. Cached on disk.

    Order of sources:
      1. cfg.local_clean_csv  (data/clean_pairs.csv, cols: passage,query)
      2. HuggingFace ms_marco (v1.1/v2.1 validation — selected passage + query)
      3. synthetic (cfg.synthetic_clean or --synthetic)
    """
    cache = os.path.join(cfg.data_dir, "clean_pairs.parquet")
    if os.path.exists(cache):
        df = pd.read_parquet(cache)
        return df.head(cfg.n_base_pairs)

    df = None
    if cfg.local_clean_csv and os.path.exists(cfg.local_clean_csv):
        raw = pd.read_csv(cfg.local_clean_csv)
        df = _normalize_clean(raw, "local_csv")
    if df is None and not cfg.synthetic_clean:
        df = _load_ms_marco(cfg)
    if df is None or len(df) == 0:
        df = _synthetic_pairs(cfg)

    df = _clean_df(df, cfg)
    os.makedirs(cfg.data_dir, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df.head(cfg.n_base_pairs)


def _normalize_clean(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    def pick(*names):
        for n in names:
            if n in raw.columns:
                return raw[n]
        return None
    p, q = pick("passage", "text", "context", "doc"), pick("query", "question", "title")
    if p is None or q is None:
        raise ValueError(f"clean pairs CSV needs 'passage' and 'query' columns; got {list(raw.columns)}")
    return pd.DataFrame({"passage": p.astype(str), "query": q.astype(str), "source_id": source})


def _pick_ms_marco_passage(ex) -> str | None:
    """Flatten HF ms_marco v1.1/v2.1 nested ``passages`` into one string."""
    raw = ex.get("passage")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    passages = ex.get("passages")
    if isinstance(passages, dict):
        texts = passages.get("passage_text") or passages.get("passage") or []
        selected = passages.get("is_selected") or []
        if texts:
            for flag, text in zip(selected, texts):
                if flag and text:
                    return str(text).strip()
            return str(texts[0]).strip() if texts[0] else None
    if isinstance(passages, list) and passages:
        p0 = passages[0]
        if isinstance(p0, dict):
            return (p0.get("passage_text") or p0.get("text") or p0.get("passage") or "")
        return str(p0).strip()
    return None


def _ms_marco_to_pairs(ds, max_rows: int) -> pd.DataFrame:
    cols = set(ds.column_names)
    qcol = "query" if "query" in cols else ("question" if "question" in cols else None)
    if qcol is None:
        return pd.DataFrame(columns=["passage", "query", "source_id"])
    rows = []
    n = min(len(ds), max(max_rows * 4, 4000))
    for i in range(n):
        if len(rows) >= max_rows:
            break
        ex = ds[i]
        q = ex.get(qcol)
        p = _pick_ms_marco_passage(ex)
        if not q or not p:
            continue
        q, p = str(q).strip(), str(p).strip()
        if len(p) > 80 and len(q) > 5:
            rows.append({"passage": p, "query": q, "source_id": "ms_marco"})
    return pd.DataFrame(rows)


def _load_ms_marco(cfg) -> pd.DataFrame:
    """Load golden query–passage pairs.

    HuggingFace ``ms_marco`` configs are ``v1.1`` / ``v2.1`` (the old
    ``passage_ranked`` config no longer exists). Passages are nested under
    ``passages.passage_text`` with ``is_selected`` flags.
    """
    from datasets import load_dataset
    need = int(getattr(cfg, "n_base_pairs", 1650) or 1650)
    attempts = [
        ("ms_marco", "v1.1", "validation"),
        ("ms_marco", "v1.1", "train"),
        ("ms_marco", "v2.1", "validation"),
        ("ms_marco", "v2.1", "train"),
        ("microsoft/ms_marco", "v1.1", "validation"),
    ]
    for repo, config, split in attempts:
        try:
            ds = load_dataset(repo, config, split=split)
            df = _ms_marco_to_pairs(ds, need)
            if len(df):
                print(f"  loaded {len(df)} pairs from {repo} {config}/{split}")
                return df
            print(f"  ms_marco ({repo}/{config}/{split}) had no usable pairs")
        except Exception as e:
            print(f"  ms_marco ({repo}/{config}/{split}) failed: {type(e).__name__}: {e}")
    print("  WARNING: ms_marco could not be loaded — falling back to synthetic pairs.")
    return pd.DataFrame(columns=["passage", "query", "source_id"])


def _synthetic_pairs(cfg) -> pd.DataFrame:
    """Deterministic offline corpus for pipeline testing (NOT for publication)."""
    topics = ["urban planning", "marine biology", "tax policy", "ancient trade routes",
              "vaccine logistics", "streaming media codecs", "soil remediation",
              "central banking", "wildfire forecasting", "library archives",
              "robotic surgery", "cryptocurrency regulation", "glacier monitoring",
              "food supply chains", "public transit design", "genetic testing",
              "cloud compute pricing", "coastal erosion", "sports analytics", "water purification"]
    sentences = [
        "The committee reviewed {t} and approved the proposed budget in March.",
        "Researchers studying {t} reported significant findings at the annual conference.",
        "Funding for {t} was doubled after the independent audit was published.",
        "A new study of {t} challenges the assumptions of earlier models.",
        "Officials said {t} would be prioritized in the next planning cycle.",
        "Public comments on {t} were collected over a 45-day period.",
        "The {t} report highlighted three areas requiring immediate attention.",
        "Experts disagree about the long-term effects of changes to {t}.",
        "The pilot program in {t} expanded to four additional regions.",
        "Data released in June provided the clearest picture of {t} to date.",
    ]
    queries = [
        "What was the main outcome of the review?",
        "Why was the budget approved?",
        "What did the researchers report?",
        "Which areas need attention?",
        "How long did the public comment period last?",
        "What did the data reveal?",
        "Where was the pilot program expanded?",
        "What do experts disagree about?",
        "What was prioritized in the planning cycle?",
        "When was the audit published?",
    ]
    rows = []
    rng = np.random.default_rng(cfg.seed)
    i = 0
    while len(rows) < cfg.n_base_pairs:
        t = topics[i % len(topics)]
        k = (i // len(topics)) % len(sentences)
        passage = " ".join(sentences[(k + j) % len(sentences)].format(t=t)
                           for j in range(4))
        # Unique suffix so drop_duplicates cannot collapse the pool to ~200
        # repeating (topic, query) templates.
        passage = f"{passage} Case file {i:04d}."
        query = queries[(i * 7 + k) % len(queries)]
        rows.append({"passage": passage, "query": query,
                     "source_id": f"synthetic-{i}"})
        i += 1
    return pd.DataFrame(rows)


def _clean_df(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["passage", "query"]).reset_index(drop=True)
    df["passage"] = df["passage"].map(lambda s: _truncate_chars(s, cfg.clean_passage_max_chars))
    df["query"] = df["query"].map(lambda s: _truncate_chars(s, cfg.clean_query_max_chars))
    # keep passages long enough for a meaningful tail region
    df = df[df["passage"].str.len() > 120].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Injection construction
# ---------------------------------------------------------------------------

def build_injection(passage: str, attack_type: str, goal: dict) -> tuple[str, tuple[int, int]]:
    """Return (injected_passage, (start, end) char offsets of the injection
    within the injected passage).

    Prefix rule: naive/fake are delimited blocks (blank line before),
    topic/combined are prose (single space before).
    """
    core = ATTACK_WRAPPERS[attack_type].format(goal=goal["text"],
                                               topic=_topic_keyword(passage))
    prefix = "\n\n" if attack_type in ("naive", "fake") else " "
    inj = prefix + core
    start = len(passage) + len(prefix)
    return passage + inj, (start, start + len(core))


def active_types(df) -> list[str]:
    """Attack types present in the built dataset (excludes clean)."""
    vals = [str(t) for t in df["attack_type"].unique() if str(t) not in ("clean", "-", "nan")]
    preferred = [t for t in ATTACK_TYPES if t in vals]
    extra = sorted(t for t in vals if t not in preferred)
    return preferred + extra


def find_qrag_jsonl(explicit: str = "") -> str:
    """Locate qrag_v1.jsonl / QuietRAG export."""
    cands = [explicit, os.environ.get("MULTI_HARM_QRAG_JSONL", "")]
    cands += [
        "data/qrag_v1.jsonl",
        "qrag_v1.jsonl",
        "data/quietrag/qrag_v1.jsonl",
    ]
    cands += glob.glob("/kaggle/input/**/qrag_v1.jsonl", recursive=True)
    cands += glob.glob("**/qrag_v1.jsonl", recursive=True)
    seen = set()
    for p in cands:
        if not p or p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            return p
    return ""


def _qrag_pick(obj: dict, *names, default=None):
    for n in names:
        if n in obj and obj[n] is not None and obj[n] != "":
            return obj[n]
    return default


def _qrag_span(obj: dict, doc: str, attack: str, label: int):
    if label == 0:
        return None, None, ""
    span = _qrag_pick(obj, "span", "injection_offset", "attack_span", "char_span")
    start = end = None
    if isinstance(span, dict):
        start = span.get("start", span.get("begin"))
        end = span.get("end")
    elif isinstance(span, (list, tuple)) and len(span) >= 2:
        start, end = span[0], span[1]
    if start is None:
        start = _qrag_pick(obj, "start", "attack_start", "inj_start")
        end = _qrag_pick(obj, "end", "attack_end", "inj_end")
    if start is not None and end is not None:
        start, end = int(start), int(end)
        payload = doc[start:end] if 0 <= start <= end <= len(doc) else ""
        if attack and payload != attack:
            # keep offsets if they round-trip; else search
            if attack in doc:
                start = doc.index(attack)
                end = start + len(attack)
                payload = attack
        return start, end, payload or attack
    if attack and attack in doc:
        start = doc.index(attack)
        return start, start + len(attack), attack
    return None, None, attack or ""


def _qrag_attack_blob(obj: dict) -> dict:
    a = obj.get("attack")
    return a if isinstance(a, dict) else {}


def _qrag_attack_type(obj: dict, label: int) -> str:
    if label == 0:
        return "clean"
    blob = _qrag_attack_blob(obj)
    at = _qrag_pick(obj, "attack_type", "harm_type", "wrapper")
    if at is None:
        at = blob.get("attack_type") or blob.get("wrapper")
    if at in ATTACK_TYPES:
        return at
    # QuietRAG nested attack: tier 1 loud, 2 quiet on-topic, 3 stealth
    tier = blob.get("tier", obj.get("tier"))
    try:
        tier = int(tier)
    except (TypeError, ValueError):
        tier = None
    fam = str(blob.get("family") or _qrag_pick(obj, "family", "attack_family", default="") or "").lower()
    vis = str(blob.get("visibility") or "").lower()
    pos = str(blob.get("position") or "").lower()
    loud = blob.get("loud_keyword_signature")
    if tier == 1:
        return "naive"
    if tier == 2:
        return "topic"
    if tier == 3:
        stealth = vis in ("hidden", "stealth", "invisible", "quiet") or pos in (
            "middle", "infix", "embedded", "mid")
        mix = ("combin" in fam or "mix" in fam or "indirect" in fam)
        if stealth or mix or loud is False:
            return "combined"
        return "fake"
    if at:
        return str(at)
    return fam or "unknown"


def load_qrag_dataset(cfg, path: str) -> pd.DataFrame:
    """Load a pre-spanned QuietRAG jsonl as the full Multi-HARM table.

    Does NOT re-wrap attacks (that would undo quiet / mid-document spans).
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tag = str(_qrag_pick(obj, "split", "role", "subset", default="")).lower()
            label = obj.get("label")
            if label is None:
                label = 0 if tag in ("negatives", "negative", "clean", "benign") else 1
            label = int(label)
            doc = str(_qrag_pick(obj, "document", "passage", "text", "context", default="") or "")
            query = str(_qrag_pick(obj, "question", "query", default="") or "")
            blob = _qrag_attack_blob(obj)
            raw_atk = obj.get("attack")
            if isinstance(raw_atk, dict):
                attack = str(raw_atk.get("text") or "")
            else:
                attack = str(_qrag_pick(obj, "attack", "injection", "payload", default="") or "")
            start, end, payload = _qrag_span(obj, doc, attack, label)
            family = str(blob.get("family") or _qrag_pick(
                obj, "family", "attack_family", "goal", default="-") or "-")
            at = _qrag_attack_type(obj, label)
            if label == 0:
                at, payload, start, end = "clean", "", None, None
            rows.append({
                "attack_type": at,
                "goal": family,
                "passage": doc,
                "query": query,
                "injection": payload,
                "injection_offset": [start, end],
                "label": label,
                "tier": obj.get("tier"),
            })
    if not rows:
        raise ValueError(f"no records in {path}")
    df = pd.DataFrame(rows)
    df["id"] = [f"{'c' if r.label == 0 else 'i'}{k:05d}" for k, r in df.iterrows()]
    f = cfg.split_frac
    shuffled = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    labels_map = {}
    for (_, _gl), grp in shuffled.groupby(["attack_type", "goal"], sort=False):
        n = len(grp)
        n1 = int(n * f[0])
        n2 = int(n * f[1])
        # keep at least 1 test row when n >= 3
        if n >= 3 and (n - n1 - n2) < 1:
            n2 = max(0, n2 - 1)
        lab = ["train"] * n1 + ["val"] * n2 + ["test"] * (n - n1 - n2)
        if not lab:
            lab = ["train"] * n
        for idx, lb in zip(grp.index, lab):
            labels_map[idx] = lb
    shuffled["split"] = [labels_map[i] for i in shuffled.index]
    n_ok = 0
    inj = shuffled[shuffled["label"] == 1]
    for _, r in inj.iterrows():
        s, e = r["injection_offset"]
        if s is None or e is None:
            continue
        if r["passage"][int(s):int(e)] == r["injection"]:
            n_ok += 1
    print(f"  QuietRAG {path}: {len(shuffled)} rows "
          f"(inj={int((shuffled.label==1).sum())} clean={int((shuffled.label==0).sum())})")
    print(f"  span round-trip OK {n_ok}/{len(inj)}")
    print(f"  types: {active_types(shuffled)}")
    print("  type counts:\n" + shuffled["attack_type"].value_counts().to_string())
    return shuffled[["id", "split", "attack_type", "goal", "passage", "query",
                     "injection", "injection_offset", "label"]]


def build_dataset(cfg) -> pd.DataFrame:
    """Full dataset (MS-MARCO wrap or QuietRAG jsonl)."""
    qpath = find_qrag_jsonl(getattr(cfg, "qrag_jsonl", "") or "")
    if qpath:
        print(f"Loading QuietRAG from {qpath} (pre-spanned, not template-wrapped) ...")
        return load_qrag_dataset(cfg, qpath)

    base = load_clean_pairs(cfg)
    rng = np.random.default_rng(cfg.seed)
    n_clean = cfg.n_clean
    n_cell = cfg.n_inj_per_cell
    n_inj_total = n_cell * len(ATTACK_TYPES) * len(GOALS)
    if len(base) < n_clean + n_inj_total:
        print(f"  WARNING: only {len(base)} base pairs available; cycling with jitter.")
    idx = np.arange(len(base))
    rng.shuffle(idx)

    rows = []
    # clean
    for i in range(n_clean):
        b = base.iloc[idx[i % len(idx)]]
        rows.append({"base_idx": int(idx[i % len(idx)]), "attack_type": "clean",
                     "goal": "-", "passage": b["passage"], "query": b["query"],
                     "injection": "", "injection_offset": [None, None], "label": 0})
    # injected: deterministic distinct base pairs per cell (no reuse across cells
    # when the pool allows it)
    pos = n_clean
    for at in ATTACK_TYPES:
        for g in GOALS:
            for k in range(n_cell):
                bi = int(idx[pos % len(idx)])
                pos += 1
                b = base.iloc[bi]
                inj_passage, off = build_injection(b["passage"], at, g)
                rows.append({"base_idx": bi, "attack_type": at, "goal": g["id"],
                             "passage": inj_passage, "query": b["query"],
                             "injection": inj_passage[off[0]:off[1]],
                             "injection_offset": list(off), "label": 1})

    df = pd.DataFrame(rows)
    df["id"] = [f"{'c' if r['label'] == 0 else 'i'}{k:05d}"
                for k, r in df.iterrows()]

    # ---- 60/20/20 stratified split by (attack_type, goal); clean is its own
    #      stratum (goal="-") -------------------------------------------------
    f = cfg.split_frac
    shuffled = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    labels_map = {}
    for (at, gl), grp in shuffled.groupby(["attack_type", "goal"], sort=False):
        n = len(grp)
        n1 = int(n * f[0])
        n2 = int(n * f[1])
        lab = ["train"] * n1 + ["val"] * n2 + ["test"] * (n - n1 - n2)
        for idx, lb in zip(grp.index, lab):
            labels_map[idx] = lb
    shuffled = shuffled.drop(columns=["base_idx"])
    shuffled["split"] = [labels_map[i] for i in shuffled.index]
    return shuffled[["id", "split", "attack_type", "goal", "passage", "query",
                     "injection", "injection_offset", "label"]]
