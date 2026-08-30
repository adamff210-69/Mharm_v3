#!/usr/bin/env python3
"""04 — Pooled H* calibration (v3 §2.1): per-head AUROC of the attention
ratio across 4 layers x 32 heads on 150-200 stratified calibration samples.
This shared head is (a) the head used by the Attention-Tracker-replication
baseline (4.8 row 1) and (b) the attention head of HARM_general (row 2).

Runs on cached signals only — seconds, not GPU time. Re-running with a
different top-K never requires another forward pass (per-head masses are
cached; see README "Design decisions").

Writes out/calib/H_star.json
"""
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from config import load_config
from multi_harm_common.calibrate import calibrate_pooled_hstar
from multi_harm_common.io_utils import load_json, save_json
from multi_harm_common.sigcache import load_cache


def main():
    cfg = load_config()
    cache = load_cache(cfg.data_dir)
    df = pd.read_parquet(os.path.join(cfg.data_dir, "dataset.parquet"))
    n_heads = max(h for (_, h) in next(iter(cache.masses.values()))) + 1
    print(f"Signals loaded: {len(cache.meta)} samples, "
          f"layers {cache.layers}, heads 0..{n_heads - 1}")

    hstar = calibrate_pooled_hstar(df, cache, cfg, n_heads)
    save_json(hstar, os.path.join(cfg.out_dir, "calib", "H_star.json"))

    print(f"\nPooled H* (n={hstar['n_clean']} clean + {hstar['n_inj']} injected "
          f"train samples):")
    print(f"  best head: layer {hstar['best_head'][0]}, head {hstar['best_head'][1]} "
          f"| AUROC {hstar['best_auroc']:.4f}")
    print(f"  top-{len(hstar['top_k'])} heads:")
    for (l, h), a in hstar["top_k"]:
        print(f"    layer {l} head {h:2d}  AUROC {a:.4f}")
    top10 = sorted(hstar["all"].items(), key=lambda kv: -kv[1])[:10]
    print("  top-10 overall:")
    for k, v in top10:
        print(f"    {k:8s} {v:.4f}")
    print(f"\nSaved -> {os.path.join(cfg.out_dir, 'calib', 'H_star.json')}")


if __name__ == "__main__":
    main()
