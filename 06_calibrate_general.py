#!/usr/bin/env python3
"""06 — 4.8 ROW 2: HARM_general (fused, shared calibration) + the
PIShield-style hidden-only shared baseline (v3 §4.8 row 2; Table B).

HARM_general: fusion of the shared attention head (H*) with a residual probe
at a single shared L*, one shared alpha and one shared theta — the
"single detector" that the per-attack specialists must beat.
PIShield-style baseline: the hidden-state probe alone (no attention, no
fusion, shared calibration) — the residual-stream side of the prior art
(Zou et al.).

Runs on cached signals only.
Writes:
  out/calib/general.json
  out/experiments/row2_general.json
  out/experiments/baseline_hidden_only.json
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import load_config
from multi_harm_common.calibrate import calibrate_specialist
from multi_harm_common.detect import score_spec_on_split
from multi_harm_common.io_utils import load_json, save_json
from multi_harm_common.metrics import auroc, type_vs_clean_ids
from multi_harm_common.sigcache import load_cache

TYPES = ["topic", "naive", "fake", "combined"]


def per_type_aurocs(scores_all: dict, cache, df, split: str):
    out = {}
    for t in TYPES:
        m = cache.subset(df[(df["split"] == split) & (df["attack_type"] == t)]["id"].tolist())
        idx = {sid: i for i, sid in enumerate(m["ids"])}
        s = np.array([scores_all[sid] for sid in m["ids"]])
        out[t] = float(auroc(m["labels"], s))
    return out


def main():
    cfg = load_config()
    hstar_path = os.path.join(cfg.out_dir, "calib", "H_star.json")
    if not os.path.exists(hstar_path):
        print("Run 04_calibrate_hstar.py first.")
        sys.exit(1)
    hstar = load_json(hstar_path)
    cache = load_cache(cfg.data_dir)
    df = pd.read_parquet(os.path.join(cfg.data_dir, "dataset.parquet"))

    print("Calibrating HARM_general (shared head + shared L*/probe/alpha/theta) ...")
    gen = calibrate_specialist("general", df, cache, cfg, hstar, use_shared_head=True)
    save_json(gen, os.path.join(cfg.out_dir, "calib", "general.json"))
    print(f"  L*={gen['L_star']} alpha={gen['alpha']:.2f} theta={gen['theta']:.4f} "
          f"| calib AUROC fused={gen['auroc']['fused']:.4f}")

    # ---- 4.8 row 2: per-type AUROC of the fused shared score (test) --------
    ev = score_spec_on_split(gen, df, cache, "test", cfg.epsilon)
    scores_all = dict(zip(ev["ids"], ev["s"]))
    per_type = per_type_aurocs(scores_all, cache, df, "test")
    row2 = {
        "config": "fused, shared calibration (HARM_general)",
        "L_star": gen["L_star"], "head": gen["head_shared"],
        "alpha": gen["alpha"], "theta": gen["theta"],
        "per_type_auroc": per_type,
        "spread": max(per_type.values()) - min(per_type.values()),
        "calib_auroc": gen["auroc"],
    }
    save_json(row2, os.path.join(cfg.out_dir, "experiments", "row2_general.json"))
    print("\n4.8 ROW 2 — fused, shared calibration (test):")
    for t in TYPES:
        print(f"  {t:10s} AUROC {per_type[t]:.4f}")
    print(f"  spread {row2['spread']:.4f}")

    # ---- PIShield-style hidden-only baseline (test) ------------------------
    from multi_harm_common.signals import probe_probs
    m = cache.subset(df[df["split"] == "test"]["id"].tolist())
    p = np.array([probe_probs(m["hid"][gen["L_star"]][i], gen["probe"])
                  for i in range(len(m["ids"]))])
    hidden_per_type = {}
    for t in TYPES:
        sub = cache.subset(type_vs_clean_ids(df, "test", t))
        idx = {sid: i for i, sid in enumerate(m["ids"])}
        s = np.array([p[idx[sid]] for sid in sub["ids"]])
        hidden_per_type[t] = float(auroc(sub["labels"], s))
    res = {
        "config": "hidden-only, shared probe at shared L* (PIShield-style baseline)",
        "L_star": gen["L_star"],
        "per_type_auroc": hidden_per_type,
        "spread": max(hidden_per_type.values()) - min(hidden_per_type.values()),
        "auroc_overall": float(auroc(m["labels"], p)),
    }
    save_json(res, os.path.join(cfg.out_dir, "experiments", "baseline_hidden_only.json"))
    print("\nPIShield-style hidden-only baseline (test):")
    for t in TYPES:
        print(f"  {t:10s} AUROC {hidden_per_type[t]:.4f}")
    print(f"  overall AUROC {res['auroc_overall']:.4f}")


if __name__ == "__main__":
    main()
