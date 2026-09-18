#!/usr/bin/env python3
"""End-to-end check of the MODEL stages (01, 03) with no network at all.

`run_offline_check.py` proves the analysis half (02, 04-11) on a fabricated
cache. This proves the half that needs a transformer: 01's shape smoke test and
AUROC self-test, 03's §2.0 token-range gate, real `forward_signals` extraction
(attention masses, clipped spans, widths), the chunked parquet writer, the
extraction checkpoint and the cache provenance record — and, with `--full`,
stages 04-11 run on genuine attention matrices from a genuine transformer
instead of planted numbers.

How it avoids the network: it builds a ~1M-parameter random-init
`LlamaForCausalLM` (so the v3.1 `num_hidden_layers` / `num_attention_heads`
lookup is exercised on the architecture family that broke v3.0) plus a real
`PreTrainedTokenizerFast`, saves both into a temp dir, and points
`MULTI_HARM_MODEL_NAME` at that directory — `from_pretrained` on a local path
needs no HuggingFace access.

The numbers are scientifically meaningless (random weights). This is a wiring
test, and it is the one worth passing before spending a Colab/Kaggle session.

    python run_tiny_model_check.py [--keep] [--full]

Needs torch + transformers (any CPU build works). If they are absent it prints
why and exits 0, so it is safe to ship in environments without them.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
N_CLEAN = 60
N_INJ_PER_CELL = 6          # 4 types x 5 goals x 6 = 120 injected; 20% of each
                            # cell lands in val, which §4.3/§4.4 need to exist
MAX_SEQ_LEN = 256


def deps_available() -> tuple[bool, str]:
    try:
        import torch                # noqa: F401
        from transformers import LlamaConfig, LlamaForCausalLM   # noqa: F401
        from tokenizers import Tokenizer                        # noqa: F401
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    return True, ""


def norm_ws(t: str) -> str:
    import re
    return re.sub(r"\s+", " ", t).strip()


def build_tiny_model(dest: str, corpus: list[str]) -> dict:
    """Random-init Llama-style LM + a real fast tokenizer, saved to `dest`."""
    from tokenizers import (Tokenizer, decoders, models,
                             pre_tokenizers, trainers)
    from transformers import (LlamaConfig, LlamaForCausalLM,
                              PreTrainedTokenizerFast)

    # A byte-level BPE, trained on the corpus: every character is encodable, so
    # nothing can fall to [UNK]. That matters more than it sounds — the §2.0 gate
    # compares decoded text against the source text, and a fixture whose tokenizer
    # loses words would make the gate "fail" for reasons that have nothing to do
    # with the pipeline. It is also a closer analogue of a real Llama tokenizer
    # than a word-level one, because subwords split mid-word, which exercises the
    # overlap>0 clipping path in _span_to_tokens instead of the easy case.
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=1200, min_frequency=1,
        special_tokens=["[UNK]", "[PAD]", "[EOS]", "<system>", "<user>",
                        "<|assistant|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
        eos_token="[EOS]")
    # The pipeline needs a chat template. Kept literal (no {% raw %}) so the role
    # markers are single vocabulary tokens, and each message's content stays a
    # contiguous substring — exactly what encode_sample's find_span relies on.
    NL = chr(10)
    fast.chat_template = ("<system>" + NL + "{% for m in messages %}<user>" + NL
                          + "{{ m['content'] }}" + NL + "{% endfor %}"
                          + "{% if add_generation_prompt %}<|assistant|>" + NL
                          + "{% endif %}")
    # Round-trip check: if encode/decode did not preserve the text, the §2.0 gate
    # would be testing the fixture instead of the pipeline. Say so here.
    probe = corpus[0] if corpus else "the quick brown fox jumps over the lazy dog."
    if norm_ws(fast.decode(fast(probe, add_special_tokens=False)["input_ids"],
                           skip_special_tokens=True)) != norm_ws(probe):
        raise SystemExit("tiny fixture is wrong: the tokenizer does not round-trip "
                         "its own corpus text, so the §2.0 gate would fail on the "
                         "fixture rather than on the pipeline")

    n_layers, n_heads = 6, 4
    cfg = LlamaConfig(vocab_size=len(tok.get_vocab()) + 8, hidden_size=64,
                      intermediate_size=128, num_hidden_layers=n_layers,
                      num_attention_heads=n_heads, num_key_value_heads=n_heads,
                      max_position_embeddings=2048,
                      attn_implementation="eager",
                      pad_token_id=fast.pad_token_id,
                      eos_token_id=fast.eos_token_id)
    model = LlamaForCausalLM(cfg)
    model.config._attn_implementation = "eager"   # SDPA returns no weights
    model.save_pretrained(dest)
    fast.save_pretrained(dest)
    return {"layers": n_layers, "heads": n_heads,
            "vocab": len(tok.get_vocab()), "path": dest}


def stage_env(work: str, model_dir: str, full: bool) -> dict:
    """Same discipline as run_offline_check.py: explicit sizes, no test mode
    (apply_test_mode clamps calibration sizes after env overrides)."""
    n_calib = 20 if full else 12
    return dict(os.environ,
                MULTI_HARM_MODEL_NAME=model_dir,
                MULTI_HARM_QUANT="fp32",
                MULTI_HARM_SYNTHETIC_CLEAN="true",
                MULTI_HARM_N_CLEAN=str(N_CLEAN),
                MULTI_HARM_N_INJ_PER_CELL=str(N_INJ_PER_CELL),
                MULTI_HARM_N_BASE_PAIRS="300",
                MULTI_HARM_MAX_SEQ_LEN=str(MAX_SEQ_LEN),
                MULTI_HARM_N_VALIDATE="8",
                MULTI_HARM_N_LATENCY_RUNS="5",
                MULTI_HARM_CALIB_PER_SPECIALIST=str(n_calib),
                MULTI_HARM_CALIB_H_SAMPLES=str(n_calib),
                MULTI_HARM_DATA_DIR=os.path.join(work, "data"),
                MULTI_HARM_OUT_DIR=os.path.join(work, "out"),
                PYTHONPATH=work,
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false")


def real_signal_checks(work: str) -> list[str]:
    """The parts only a real forward pass can prove."""
    bad: list[str] = []
    import numpy as np
    import pandas as pd
    sys.path.insert(0, work)
    os.chdir(work)
    from multi_harm_common import sigcache as SC
    from multi_harm_common.signals import head_ratio

    df = pd.read_parquet(os.path.join(work, "data", "dataset.parquet"))
    rep_path = os.path.join(work, "out", "validation", "extraction_report.json")
    rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    if not rep:
        bad.append("03 wrote no extraction_report.json")
    cache = SC.load_cache(os.path.join(work, "data"))
    have = len(cache.masses)
    if have != len(df) - int(rep.get("n_skipped", 0)):
        bad.append(f"cache has {have} rows, dataset has {len(df)} with "
                   f"{rep.get('n_skipped')} skipped")
    else:
        print(f"  cache: {have}/{len(df)} rows (skipped {rep.get('n_skipped')}) OK")
    prov = SC.load_provenance(os.path.join(work, "data"))
    for k in ("model_name", "quant", "n_model_layers", "n_skipped",
              "dataset_fingerprint", "max_seq_len"):
        if k not in prov:
            bad.append(f"sigcache provenance missing '{k}': {sorted(prov)}")
    if prov.get("n_model_layers") != 6:
        bad.append(f"provenance recorded {prov.get('n_model_layers')} layers, the "
                   f"tiny model has 6 — the LlamaConfig was misread (P0-2)")
    else:
        print("  provenance read num_hidden_layers=6 off a LlamaConfig OK")

    m = cache.subset(df["id"].tolist())
    if len(m["ids"]) != have:
        bad.append(f"subset returned {len(m['ids'])} of {have}")
    widths = np.asarray(m["widths"], dtype=float)
    if (widths <= 0).any():
        bad.append("non-positive span widths in the real cache")
    if (widths[:, 1] > widths[:, 0]).any():
        bad.append("injection span wider than its passage — clipping is "
                   "inconsistent between chat.encode_sample and "
                   "model.forward_signals")

    # The masses are per-head sums of the attention row over three spans. The
    # injection span is a subset of the passage span, so m_qi <= m_qp must hold
    # for EVERY head — a span/clip mismatch anywhere breaks that before it breaks
    # a metric. Also checks the cached grid shape and that R is finite.
    keys = sorted(next(iter(m["masses"])).keys())
    want = int(prov.get("attn_last_k", 0)) * int(prov.get("n_model_heads", 0))
    if len(keys) != want:
        bad.append(f"cached {len(keys)} (layer,head) mass rows per sample; the "
                   f"model is {prov.get('n_model_layers')} layers x "
                   f"{prov.get('n_model_heads')} heads and signals come from the "
                   f"last {prov.get('attn_last_k')} layers = {want} rows")
    r_all = []
    for j in range(len(m["ids"])):
        mm = np.array([m["masses"][j][k] for k in keys], dtype=np.float64)
        if mm.shape != (len(keys), 3):
            bad.append(f"mass row shape {mm.shape}, expected ({len(keys)}, 3)")
            break
        if not np.isfinite(mm).all() or (mm < -1e-6).any():
            bad.append(f"non-finite/negative masses for {m['ids'][j]}")
            break
        if (mm[:, 1] > mm[:, 0] + 1e-6).any():
            bad.append(f"{m['ids'][j]}: m_qi > m_qp on some head, so the "
                       f"injection span is not inside the passage span")
            break
        r_all.append(head_ratio(mm[0], 1e-9, tuple(int(x) for x in widths[j])))
    if not bad:
        r_all = np.array(r_all)
        print(f"  masses ({len(keys)}x3) per sample: m_qi <= m_qp for all "
              f"{len(m['ids'])} samples OK")
        print(f"  R over real signals: mean={r_all.mean():.4f} "
              f"p95={np.percentile(r_all, 95):.4f} max={r_all.max():.4f}")
    return bad


def _run(work: str, env: dict, script: str, extra: list[str]) -> str:
    print(f"\n--- {script} {' '.join(extra)} ---")
    r = subprocess.run([sys.executable, script] + extra, cwd=work, env=env,
                       text=True, capture_output=True, timeout=3600)
    out = (r.stdout or "").strip()
    print("\n".join(out.splitlines()[-14:]))
    if r.returncode != 0:
        print("STDERR:\n" + (r.stderr or "")[-4000:])
        raise SystemExit(f"{script} exited {r.returncode}")
    if "Traceback" in (r.stderr or ""):
        print("STDERR:\n" + (r.stderr or "")[-2000:])
        raise SystemExit(f"{script} printed a traceback")
    if "SKIP" in out or "skipped" in out:
        for ln in out.splitlines():
            if "SKIP" in ln or "skipped" in ln:
                print("   >", ln.strip())
    return out


def guard_checks(work: str, env: dict) -> list[str]:
    """Exercise the two v3.1 cache guards for real, on a shrunk dataset:
    a foreign cache must abort, and --fresh must rebuild it."""
    bad: list[str] = []
    sig = os.path.join(work, "data", "signals")
    fp = os.path.join(work, "data", "dataset_fingerprint.json")
    snap, snap_fp = sig + "_snap", fp + "_snap"
    if os.path.isdir(sig):
        shutil.copytree(sig, snap)
    if os.path.exists(fp):
        shutil.copy2(fp, snap_fp)
    try:
        env2 = dict(env, MULTI_HARM_N_CLEAN="16")
        _run(work, env2, "02_build_dataset.py", [])        # rewrites dataset+fp
        os.makedirs(sig, exist_ok=True)
        shutil.rmtree(sig, ignore_errors=True)
        shutil.copytree(snap, sig)                          # foreign cache, new fp
        print("\n--- 03 (expect: refuses a foreign cache) ---")
        r = subprocess.run([sys.executable, "03_extract_signals.py"], cwd=work,
                           env=env2, text=True, capture_output=True, timeout=900)
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode == 0:
            bad.append("03 accepted a signal cache built from a DIFFERENT dataset; "
                       "the checkpoint would mark the new ids done and 04-11 would "
                       "silently run on a shrunken set")
        elif "DIFFERENT dataset" not in out:
            bad.append(f"03 aborted for the wrong reason:\n{out[-800:]}")
        else:
            print("   refused, naming the changed fields  OK")
            for ln in out.splitlines():
                if "changed:" in ln:
                    print("   >", ln.strip()[:150])
                    break
        _run(work, env2, "03_extract_signals.py", ["--fresh"])
        import pandas as pd
        df2 = pd.read_parquet(os.path.join(work, "data", "dataset.parquet"))
        prov = json.load(open(os.path.join(sig, "sigcache_meta.json")))
        if int(prov.get("n_rows", -1)) != len(df2):
            bad.append(f"--fresh rebuilt a cache for {prov.get('n_rows')} rows but "
                       f"the dataset has {len(df2)}")
        else:
            print(f"  --fresh rebuilt the cache for the new dataset "
                  f"({len(df2)} rows) OK")
    finally:
        shutil.rmtree(snap, ignore_errors=True)
        if os.path.exists(snap_fp):
            os.remove(snap_fp)
    return bad


def main() -> int:
    ok, why = deps_available()
    if not ok:
        print(f"run_tiny_model_check.py: SKIPPED (torch/transformers unavailable: "
              f"{why})\nThis check covers stages 01 and 03, the two that need a "
              f"model. run_offline_check.py covers the rest without them.")
        return 0
    keep = "--keep" in sys.argv
    full = "--full" in sys.argv or "--keep" in sys.argv
    work = tempfile.mkdtemp(prefix="mh_tiny_")
    for f in os.listdir(ROOT):
        if f.endswith(".py") or f in ("multi_harm_common", "requirements.txt",
                                      "config.py"):
            src = os.path.join(ROOT, f)
            (shutil.copytree(src, os.path.join(work, f),
                             ignore=shutil.ignore_patterns("__pycache__"))
             if os.path.isdir(src) else shutil.copy(src, os.path.join(work, f)))
    model_dir = os.path.join(work, "_tinymodel")
    env = stage_env(work, model_dir, full)
    bad: list[str] = []
    code = 0
    try:
        # 02 first: it needs no model, and its texts define the tokenizer vocab
        _run(work, env, "02_build_dataset.py", ["--synthetic"])
        import pandas as pd
        df = pd.read_parquet(os.path.join(work, "data", "dataset.parquet"))
        cols = [c for c in ("passage", "query", "injection_text", "goal_text")
                if c in df.columns]
        corpus = [" ".join(str(r[c]) for c in cols) for r in df.to_dict("records")]
        for s_ in df.to_dict("records"):
            if s_.get("passage"):
                corpus.append(str(s_["passage"]))
            if s_.get("query"):
                corpus.append(str(s_["query"]))
        info = build_tiny_model(model_dir, corpus)
        print(f"\ntiny model: {info['layers']} layers x {info['heads']} heads, "
              f"vocab {info['vocab']}, saved to {info['path']}")

        _run(work, env, "01_setup_and_validate.py", [])
        _run(work, env, "03_extract_signals.py", [])
        bad += real_signal_checks(work)

        if full:
            for script, extra in [("04_calibrate_hstar.py", []),
                                  ("05_baseline_attn_tracker.py", []),
                                  ("06_calibrate_general.py", []),
                                  ("07_calibrate_specialists.py", []),
                                  ("08_meta_decision.py", []),
                                  ("09_experiments_analysis.py", ["--calib-sweep"]),
                                  ("10_figures_report.py", []),
                                  ("11_reproducibility.py", [])]:
                _run(work, env, script, extra)
            for rel in ("out/experiments/table_48.json",
                        "out/experiments/span_width.json",
                        "out/experiments/sec43_head_by_type.json",
                        "out/validation/width_invariance.json",
                        "out/report/RESULTS.md", "out/repro/repro_manifest.json"):
                if not os.path.exists(os.path.join(work, rel)):
                    bad.append(f"no {rel} after a real-model run")
            man = json.load(open(os.path.join(
                work, "out", "repro", "repro_manifest.json")))
            if not man.get("artifact_sha256"):
                bad.append("repro_manifest.json hashed nothing")
            print(f"  04-11 ran on real attention signals; "
                  f"{len(man.get('artifact_sha256', {}))} artifacts hashed")
        bad += guard_checks(work, env)
    except SystemExit as e:
        print(f"\nTINY-MODEL CHECK FAILED: {e}")
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        bad.append(f"crashed: {type(e).__name__}: {e}")

    if bad:
        print("\nTINY-MODEL CHECK FAILED:")
        for b in bad:
            print(f"  - {b}")
        code = 1
    else:
        print("\nTINY-MODEL CHECK: ALL PASS  (stages 01/03 verified against a real "
              "transformer forward pass; not a scientific result — random weights)")
    print(f"  work dir: {work}" + ("" if keep else " (removed)"))
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
