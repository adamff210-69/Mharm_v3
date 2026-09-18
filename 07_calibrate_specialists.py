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
from config import load_config, ATTACK_TYPES
from multi_harm_common.calibrate import calibrate_specialist
from multi_harm_common.io_utils import load_json, save_json
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

    specs = {}
    for t in TYPES:
        print(f"Calibrating specialist: {t} ...")
        sp = calibrate_specialist(t, df, cache, cfg, hstar, use_shared_head=False)
        specs[t] = sp
        a = sp["auroc"]
        print(f"  L*={sp['L_star']}  head=({sp['head'][0]},{sp['head'][1]})  "
              f"alpha={sp['alpha']:.2f}  theta={sp['theta']:.4f}")
        ti = sp["calib"]["theta_info"]
        if ti.get("fpr_resolution_limited"):
            print(f"  NOTE: {ti['n_neg']} clean calibration samples -> the smallest "
                  f"non-zero FPR is {ti['min_nonzero_fpr']:.3f}, above the budget "
                  f"{sp['fpr_budget']:.4f}. theta is therefore the max clean score "
                  f"(one order statistic); expect test FPR to differ from target. "
                  f"Needs >= {int(1 / max(sp['fpr_budget'], 1e-9)) + 1} clean "
                  f"samples to resolve this budget.")
        if not ti.get("feasible", True):
            print(f"  WARNING: no threshold met the FPR budget {sp['fpr_budget']:.4f} "
                  f"on {ti.get('n_neg')} clean calibration samples (1 FP alone "
                  f"exceeds it) -> theta is 'never fire' and this type can only be "
                  f"detected via HARM_general. Raise "
                  f"MULTI_HARM_CALIB_PER_SPECIALIST above ~4/target_fpr clean "
                  f"samples ({int(4 / cfg.target_fpr) + 1}+).")
        print(f"  HALF SPLIT (v3 Phase 3 addition, calib AUROC): "
              f"attention={a['att_head']:.4f}  hidden={a['hid']:.4f}  "
              f"fused={a['fused']:.4f}  (shared-head attn={a['att_shared']:.4f})")
        gain_f = a["fused"] - a["att_head"]
        print(f"  fusion gain over attention half: {gain_f:+.4f}")

    save_json(specs, os.path.join(cfg.out_dir, "calib", "specialists.json"))

    # ---- the table the paper's §5 figure is built from ----------------------
    print("\n" + "=" * 92)
    print("HALF-SPLIT AUROC TABLE — CALIBRATION SET (per specialist)")
    print("=" * 92)
    print(f"{'specialist':12s} {'L*':>4s} {'head':>9s} {'alpha':>6s} "
          f"{'att-half':>9s} {'hid-half':>9s} {'fused':>7s} {'att-gain':>9s} "
          f"{'fused*':>7s}")
    for t in TYPES:
        sp = specs[t]
        a = sp["auroc"]
        fe = a.get("fused_eval")
        print(f"{t:12s} {sp['L_star']:>4d} ({sp['head'][0]:>2d},{sp['head'][1]:>2d}) "
              f"{sp['alpha']:>6.2f} {a['att_head']:>9.4f} {a['hid']:>9.4f} "
              f"{a['fused']:>7.4f} {a['fused'] - a['att_head']:>+9.4f} "
              + (f"{fe:>7.4f}" if isinstance(fe, float) else f"{'n/a':>7s}"))
    print("\n* fused* = fused AUROC on the held-out fraction of the calibration set")
    print("  (the rows the probe was NOT fit on). 'fused' itself is OPTIMISTIC:")
    print("  alpha is grid-searched to maximize exactly that number, so fused >="
          "max(att,hid) holds on it by construction. Quote the * column, or the")
    print("  test-split numbers from 09 (§4.8), in the paper.")
    print("\nInterpretation (v3 narrowed hypothesis): if att-half AUROCs are flat")
    print("across specialists while hid-half AUROCs vary and drive the fused gains,")
    print("specialization is carried by the residual-stream half — the paper's")
    print("central empirical finding. A null here is a result, not a failure")
    print("(run 09 --calib-sweep to check the gain is not just calibration size).")
    print(f"\nSaved -> {os.path.join(cfg.out_dir, 'calib', 'specialists.json')}")


if __name__ == "__main__":
    main()
