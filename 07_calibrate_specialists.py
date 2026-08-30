#!/usr/bin/env python3
"""07 — Per-attack-type specialist calibration (v3 Phase 3).

For each of the 4 attack types: L* -> probe -> h_base -> alpha -> theta,
with the v3 Phase 3 addition: the attention-half AUROC and the
hidden-state-half (P(injection)) AUROC are logged separately, BEFORE
fusion, for every specialist. That half-split table is the direct test of
the narrowed central hypothesis (specialization should be carried by the
residual-stream half, if at all).

Runs on cached signals only. Writes out/calib/specialists.json
"""
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from config import load_config
from multi_harm_common.calibrate import calibrate_specialist
from multi_harm_common.io_utils import load_json, save_json
from multi_harm_common.sigcache import load_cache

TYPES = ["topic", "naive", "fake", "combined"]


def main():
    cfg = load_config()
    hstar_path = os.path.join(cfg.out_dir, "calib", "H_star.json")
    if not os.path.exists(hstar_path):
        print("Run 04_calibrate_hstar.py first.")
        sys.exit(1)
    hstar = load_json(hstar_path)
    cache = load_cache(cfg.data_dir)
    df = pd.read_parquet(os.path.join(cfg.data_dir, "dataset.parquet"))

    specs = {}
    for t in TYPES:
        print(f"Calibrating specialist: {t} ...")
        sp = calibrate_specialist(t, df, cache, cfg, hstar, use_shared_head=False)
        specs[t] = sp
        a = sp["auroc"]
        print(f"  L*={sp['L_star']}  head=({sp['head'][0]},{sp['head'][1]})  "
              f"alpha={sp['alpha']:.2f}  theta={sp['theta']:.4f}")
        print(f"  HALF SPLIT (v3 Phase 3 addition, calib AUROC): "
              f"attention={a['att_head']:.4f}  hidden={a['hid']:.4f}  "
              f"fused={a['fused']:.4f}  (shared-head attn={a['att_shared']:.4f})")
        gain_f = a["fused"] - a["att_head"]
        print(f"  fusion gain over attention half: {gain_f:+.4f}")

    save_json(specs, os.path.join(cfg.out_dir, "calib", "specialists.json"))

    # ---- the table the paper's §5 figure is built from ----------------------
    print("\n" + "=" * 78)
    print("HALF-SPLIT AUROC TABLE (per specialist, calibration set)")
    print("=" * 78)
    print(f"{'specialist':12s} {'L*':>4s} {'head':>9s} {'alpha':>6s} "
          f"{'att-half':>9s} {'hid-half':>9s} {'fused':>7s} {'att-gain':>9s}")
    for t in TYPES:
        sp = specs[t]
        a = sp["auroc"]
        print(f"{t:12s} {sp['L_star']:>4d} ({sp['head'][0]:>2d},{sp['head'][1]:>2d}) "
              f"{sp['alpha']:>6.2f} {a['att_head']:>9.4f} {a['hid']:>9.4f} "
              f"{a['fused']:>7.4f} {a['fused'] - a['att_head']:>+9.4f}")
    print("\nInterpretation (v3 narrowed hypothesis): if att-half AUROCs are flat")
    print("across specialists while hid-half AUROCs vary and drive the fused gains,")
    print("specialization is carried by the residual-stream half — the paper's")
    print("central empirical finding.")
    print(f"\nSaved -> {os.path.join(cfg.out_dir, 'calib', 'specialists.json')}")


if __name__ == "__main__":
    main()
