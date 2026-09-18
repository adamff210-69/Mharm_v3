#!/usr/bin/env python3
"""05 — 4.8 ROW 1: Attention-Tracker-replication baseline (v3 §4.8).

The attention-ratio signal ALONE (no fusion, no specialization) with ONE
shared calibration (shared head H*, shared z-stats, shared threshold) across
all 4 attack types — exactly the operating mode reported in Attention
Tracker (Hung et al.). This is the reference point every other row in the
§4.8 table is compared against, and it directly engages Attention Tracker's
claim that its signal generalizes across attack types.

Per-type number = AUROC of R on {injected of that type} ∪ {clean} of the test
split (detect.type_vs_clean_ids). An AUROC is a two-class statistic; the
v3.0 version filtered to the type alone, which made every cell a constant
0.5000. Rows 2 and 3 use the same helper, so the three rows stay comparable.

Runs on cached signals only. Writes out/experiments/row1_attn_shared.json
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import load_config, ATTACK_TYPES
from multi_harm_common import signals as S
from multi_harm_common.detect import per_type_auroc, spread_of
from multi_harm_common.io_utils import load_json, save_json
from multi_harm_common.metrics import auroc, tpr_fpr
from multi_harm_common.sigcache import load_cache, usable_df

TYPES = ATTACK_TYPES


def main():
    cfg = load_config()
    hstar_path = os.path.join(cfg.out_dir, "calib", "H_star.json")
    if not os.path.exists(hstar_path):
        print("Run 04_calibrate_hstar.py first.")
        sys.exit(1)
    hstar = load_json(hstar_path)
    cache = load_cache(cfg.data_dir)
    df = usable_df(pd.read_parquet(os.path.join(cfg.data_dir, "dataset.parquet")),
                   cache)
    head = tuple(hstar["best_head"])
    print(f"Shared head: layer {head[0]} head {head[1]} "
          f"(pooled AUROC {hstar['best_auroc']:.4f})")

    # ---- shared calibration (train split, same set as H*) -----------------
    sub = cache.subset(hstar["calib_ids"])
    r_cal = np.array([S.head_ratio(m[head], cfg.epsilon, sub["widths"][i])
                      for i, m in enumerate(sub["masses"])])
    zr, mu, sd = S.zscore(r_cal)
    theta = S.choose_theta(zr, sub["labels"], cfg.target_fpr)["theta"]
    print(f"Shared threshold theta={theta:.4f} (FPR budget {cfg.target_fpr})")

    # ---- evaluate on test ---------------------------------------------------
    m = cache.subset(df[df["split"] == "test"]["id"].tolist())
    r_all = np.array([S.head_ratio(x[head], cfg.epsilon, m["widths"][i])
                      for i, x in enumerate(m["masses"])])
    z_all = (r_all - mu) / max(sd, 1e-12)
    per_type = per_type_auroc(dict(zip(m["ids"], r_all)), cache, df, "test",
                              TYPES)
    tpr, fpr = tpr_fpr(m["labels"], z_all, theta)
    counts = {}
    for t in TYPES:
        sel = df[(df["split"] == "test") &
                 (df["attack_type"].isin([t, "clean"]))]
        counts[t] = {"injected": int((sel["attack_type"] == t).sum()),
                     "clean": int((sel["attack_type"] == "clean").sum())}
    res = {
        "config": "attention-only, shared calibration (replicates Attention Tracker)",
        "head": list(head), "theta": float(theta),
        "per_type_auroc": per_type,
        "per_type_counts": counts,
        "spread": spread_of(per_type),
        "auroc_overall": float(auroc(m["labels"], r_all)),
        "detection": float(tpr), "asr": float(1 - tpr), "fpr": float(fpr),
        "n_test": len(m["ids"]),
        "eval_set": "per type: injected-of-type + all clean rows of test split",
    }
    save_json(res, os.path.join(cfg.out_dir, "experiments", "row1_attn_shared.json"))
    print("\n4.8 ROW 1 — attention-only, shared calibration (test):")
    for t in TYPES:
        a = per_type[t]
        print(f"  {t:10s} AUROC {a:.4f}" if a is not None else f"  {t:10s} n/a")
    print(f"  spread {res['spread']:.4f} | overall AUROC {res['auroc_overall']:.4f} "
          f"| ASR {res['asr']:.4f} | FPR {res['fpr']:.4f}")
    if res["spread"] < 0.03:
        print("  -> low spread: consistent with Attention Tracker's cross-attack "
              "generalization claim.")
    else:
        print(f"  -> spread {res['spread']:.3f}: the shared attention signal shows "
              "per-type variation in THIS setting (report this in §4.8).")
    if all(v is not None for v in per_type.values()):
        print("  (single-class guard: all four per-type sets contain both classes.)")
    else:
        print("  WARNING: a per-type set was degenerate — check that 02 emitted "
              "clean rows for this split.")


if __name__ == "__main__":
    main()
