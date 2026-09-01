#!/usr/bin/env python3
"""03 — Shared forward pass + signal extraction (v3 Phase 2), with:

* §2.0 gate: token-range validation on ~12 samples BEFORE any extraction;
  the script aborts (non-zero exit) if a single range check fails.
* resumable, chunked extraction (v3 §2.2): parquet appended in chunks of
  `chunk_rows` (200); a checkpoint file lets a restarted Colab session skip
  already-extracted samples.
* `--quant-compare`: v3 §2.3 — run the same N samples under the best
  available reference dtype (fp16 -> bf16 -> int8 chain; on a T4 fp16 8B
  usually OOMs, int8 is the realistic reference) and under the deployment
  dtype (nf4), and report per-head R correlations and per-layer hidden-state
  cosine fidelity. Saves out/experiments/quant_compare.json.

Run:
  python 03_extract_signals.py                  # full extraction (default)
  python 03_extract_signals.py --limit 8        # smoke: 8 mixed rows then STOP
  python 03_extract_signals.py --quant-compare  # §2.3 comparison only
  python 03_extract_signals.py --validate-only  # just re-run the §2.0 gate
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from config import load_config
from multi_harm_common import env as ENV
from multi_harm_common import signals as S
from multi_harm_common.chat import encode_sample, validate_token_ranges
from multi_harm_common.io_utils import Checkpoint, save_json, ensure_dir
from multi_harm_common.metrics import auroc, pearson
from multi_harm_common.model import forward_signals, get_n_layers, load_model
from multi_harm_common.sigcache import set_data_dir, save_row

sys.stdout.reconfigure(line_buffering=True)


# ===========================================================================
# §2.0 — token-range validation gate
# ===========================================================================

def run_validation_gate(cfg, tokenizer, df):
    print("\n=== §2.0 TOKEN-RANGE VALIDATION GATE ===")
    rng = np.random.default_rng(cfg.seed)
    picks = []
    for t in ["clean", "topic", "naive", "fake", "combined"]:
        sub = df[df["attack_type"] == t]
        picks.extend(sub.sample(min(3, len(sub)), random_state=cfg.seed)["id"].tolist())
    picks = picks[:cfg.n_validate]
    samples = df[df["id"].isin(picks)].to_dict("records")
    report = validate_token_ranges(tokenizer, samples, cfg.max_seq_len, cfg.tail_len)
    ensure_dir(os.path.join(cfg.out_dir, "validation"))
    save_json(report, os.path.join(cfg.out_dir, "validation", "token_ranges_report.json"))
    for r in report["results"]:
        marks = " ".join(f"{k}:{'OK' if v else 'FAIL'}"
                         for k, v in r.get("checks", {}).items())
        print(f"  {r['id']:8s} {r['attack_type']:9s} tokens={r['n_tokens']:4d} "
              f"passage={r['passage_range']} query={r['query_range']} "
              f"inj={r['inj_range']} | {marks} {r.get('note', '')}")
    print(f"  -> {report['n_ok']}/{report['n_samples']} passed")
    if not report["passed"]:
        print("\nFATAL: token-range validation FAILED. A misaligned range would make "
              "every downstream AUROC meaningless (v3 §2.0). Inspect "
              "out/validation/token_ranges_report.json and fix the prompt layout "
              "in multi_harm_common/chat.py before proceeding.")
        sys.exit(1)
    print("  GATE PASSED — proceeding with extraction.\n")


# ===========================================================================
# Full extraction
# ===========================================================================

def extract_all(cfg, model, tokenizer, df):
    n_layers = get_n_layers(model)
    cand = cfg.candidate_layers(n_layers)
    set_data_dir(cfg.data_dir)
    ck = Checkpoint(os.path.join(cfg.out_dir, "progress", "extract.json"))
    todo = [s for s in df.to_dict("records") if not ck.done(s["id"])]
    print(f"Extraction: {len(df) - len(todo)} done, {len(todo)} to go "
          f"(chunk={cfg.chunk_rows})")
    if ck.finished and not todo:
        print("Extraction already finished (checkpoint says so).")
        return

    t0 = time.time()
    for i, s in enumerate(todo):
        enc = encode_sample(tokenizer, s, cfg.max_seq_len, cfg.tail_len)
        if not enc.valid:
            print(f"  SKIP {s['id']}: {enc.note}")
            ck.mark_done(s["id"])
            continue
        sig = forward_signals(model, enc, cfg.attn_last_k, cand)
        save_row(cfg.data_dir, {
            "sample_id": s["id"], "split": s["split"],
            "attack_type": s["attack_type"], "goal": s["goal"],
            "label": int(s["label"]),
            "masses": sig["masses"],
            "widths": sig["widths"],
            "hidden": {l: v for l, v in sig["hidden"].items()},
        })
        ck.mark_done(s["id"])
        if (i + 1) % 25 == 0:
            ck.save()
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (len(todo) - i - 1)
            print(f"  {i + 1}/{len(todo)} | {el / 60:.1f} min elapsed | "
                  f"ETA {eta / 60:.1f} min")
    ck.finish()
    print(f"Extraction complete in {(time.time() - t0) / 60:.1f} min "
          f"(checkpoint: out/progress/extract.json)")


# ===========================================================================
# §2.3 — quantization comparison
# ===========================================================================

def _load_model_for(cfg, quant: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = ENV.pick_device()
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    bnb = ENV.bnb_config_for(quant)
    if bnb is not None:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, quantization_config=bnb, trust_remote_code=True,
            attn_implementation="eager").to("cuda")
    else:
        dtype = ENV.dtype_for(quant, device)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=dtype, trust_remote_code=True,
            attn_implementation="eager").to(device)
    model.eval()
    return model, tok, device


def _free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_subset(cfg, df, ids, quant):
    model, tok, device = _load_model_for(cfg, quant)
    n_layers = get_n_layers(model)
    cand = cfg.candidate_layers(n_layers)
    rows = df[df["id"].isin(ids)].to_dict("records")
    out = []
    for s in rows:
        enc = encode_sample(tok, s, cfg.max_seq_len, cfg.tail_len)
        sig = forward_signals(model, enc, cfg.attn_last_k, cand)
        out.append({"id": s["id"], "label": int(s["label"]),
                    "attack_type": s["attack_type"],
                    "masses": sig["masses"],
                    "widths": sig["widths"],
                    "hidden": sig["hidden"]})
    _free(model)
    return out


def run_quant_compare(cfg, df):
    print("\n=== §2.3 QUANTIZATION COMPARISON (fp-ref vs deployment) ===")
    ref_chain = [q for q in cfg.quant_compare_ref_list() if q != "nf4"]
    if ENV.pick_device() == "cpu":
        print("CPU device: quantization comparison not applicable "
              "(deployment dtype is fp32 here). Skipping.")
        save_json({"applicable": False, "reason": "cpu"},
                  os.path.join(cfg.out_dir, "experiments", "quant_compare.json"))
        return

    rng = np.random.default_rng(cfg.seed)
    ids = []
    for t in ["clean", "topic", "naive", "fake", "combined"]:
        sub = df[df["attack_type"] == t]
        ids.extend(sub.sample(min(cfg.quant_compare_n // 5, len(sub)),
                              random_state=cfg.seed)["id"].tolist())
    ids = ids[:cfg.quant_compare_n]
    print(f"  subset: {len(ids)} samples (stratified across types)")

    ref = None
    ref_used = None
    for q in ref_chain:
        try:
            ref = extract_subset(cfg, df, ids, q)
            ref_used = q
            print(f"  reference dtype: {q} (loadable on this device)")
            break
        except Exception as e:
            print(f"  {q} not usable: {type(e).__name__}: {e} — trying next ...")
    if ref is None:
        print("  No reference dtype loadable; skipping comparison.")
        save_json({"applicable": False, "reason": "no reference dtype loadable"},
                  os.path.join(cfg.out_dir, "experiments", "quant_compare.json"))
        return

    print("  deployment dtype: nf4 ...")
    dep = extract_subset(cfg, df, ids, "nf4")
    rmap = {r["id"]: r for r in ref}
    dmap = {r["id"]: r for r in dep}

    heads = sorted({(l, h) for r in ref for (l, h) in r["masses"]})
    labels = np.array([rmap[i]["label"] for i in ids])
    head_corr, head_auroc_ref = {}, {}
    for (l, h) in heads:
        # per-column-mean (span-normalized) ratio, identical definition in
        # both dtypes (see signals.head_ratio)
        a = np.array([S.head_ratio(rmap[i]["masses"][(l, h)], cfg.epsilon,
                                   rmap[i]["widths"]) for i in ids])
        b = np.array([S.head_ratio(dmap[i]["masses"][(l, h)], cfg.epsilon,
                                   dmap[i]["widths"]) for i in ids])
        head_corr[f"{l}x{h}"] = pearson(a, b)
        head_auroc_ref[f"{l}x{h}"] = auroc(labels, a)
    best = max(head_auroc_ref, key=head_auroc_ref.get)
    pooled = head_corr[best]

    per_layer_cos = {}
    layers = sorted({l for r in ref for l in r["hidden"]})
    for l in layers:
        cs = []
        for i in ids:
            a = rmap[i]["hidden"][l].astype(np.float32)
            b = dmap[i]["hidden"][l].astype(np.float32)
            cs.append(float(np.dot(a, b) /
                            (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        per_layer_cos[l] = float(np.mean(cs))

    res = {"applicable": True, "reference": ref_used, "n_samples": len(ids),
           "ids": ids, "best_head_ref": best,
           "pooled_r_corr_at_best_head": float(pooled),
           "mean_r_corr_all_heads": float(np.mean(list(head_corr.values()))),
           "min_r_corr_all_heads": float(np.min(list(head_corr.values()))),
           "per_head_corr": head_corr, "per_head_auroc_ref": head_auroc_ref,
           "per_layer_cos": per_layer_cos,
           "criterion": "> 0.9 at chosen L* (checked by 10_figures_report.py)"}
    ensure_dir(os.path.join(cfg.out_dir, "experiments"))
    save_json(res, os.path.join(cfg.out_dir, "experiments", "quant_compare.json"))
    print(f"  best ref head {best}: corr={pooled:.4f} | mean over heads="
          f"{res['mean_r_corr_all_heads']:.4f}")
    print(f"  per-layer hidden cosine: "
          + ", ".join(f"L{l}={v:.3f}" for l, v in per_layer_cos.items()))
    print("  Saved out/experiments/quant_compare.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-compare", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: extract at most N mixed (inj+clean) rows")
    ap.add_argument("--include-clean", action="store_true",
                    help="kept for runbooks; extraction already includes clean rows")
    args = ap.parse_args()

    cfg = load_config()
    ENV.print_env(cfg, ENV.pick_device(), ENV.resolve_quant(cfg.quant, ENV.pick_device()))

    import os.path
    if not os.path.exists(os.path.join(cfg.data_dir, "dataset.parquet")):
        print("dataset.parquet missing — run 02_build_dataset.py first.")
        sys.exit(1)
    df = pd.read_parquet(os.path.join(cfg.data_dir, "dataset.parquet"))
    if args.limit and args.limit > 0:
        inj = df[df["label"] == 1]
        cln = df[df["label"] == 0]
        n = min(int(args.limit), len(df))
        n_cln = min(len(cln), max(1, n // 4)) if len(cln) else 0
        n_inj = min(len(inj), n - n_cln)
        parts = []
        if n_inj:
            parts.append(inj.sample(n_inj, random_state=cfg.seed))
        if n_cln:
            parts.append(cln.sample(n_cln, random_state=cfg.seed))
        df = pd.concat(parts).sample(frac=1.0, random_state=cfg.seed)
        print(f"  --limit {n}: using {len(df)} rows "
              f"(inj={int((df.label==1).sum())} clean={int((df.label==0).sum())})")

    model, tokenizer, device, quant = load_model(cfg)
    if args.validate_only:
        run_validation_gate(cfg, tokenizer, df)
        return
    run_validation_gate(cfg, tokenizer, df)
    if args.quant_compare:
        _free(model)
        run_quant_compare(cfg, df)
        return
    extract_all(cfg, model, tokenizer, df)


if __name__ == "__main__":
    main()
