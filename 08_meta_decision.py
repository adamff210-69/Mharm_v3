#!/usr/bin/env python3
"""08 — Meta-decision layer (v3 architecture: 5 specialist scores ->
INJECTED/SAFE + attribution).

Evaluates the meta layer on the VAL split (train was used for all
calibration; val is the tuning split for anything meta-level):

* default mode `per_spec`: OR over the 4 individually-thresholded type
  specialists (each theta calibrated at FPR budget = 5%/4 by union bound),
  HARM_general as fallback when no type specialist fires;
* alternative mode `global_max`: single threshold on the max specialist
  score, chosen on val (reported here, applied to test in 09).

Checks the v3 FPR < 5% criterion. Writes:
  out/meta/val_eval.json            (per_spec, the default)
  out/meta/meta_config.json         (mode + global_max theta)
  out/meta/val_records.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, ".")
from config import load_config
from multi_harm_common.detect import evaluate_meta, score_spec_on_split
from multi_harm_common.io_utils import load_json, save_json
from multi_harm_common.signals import choose_theta
from multi_harm_common.sigcache import load_cache


def main():
    cfg = load_config()
    from multi_harm_common.io_utils import ensure_dir
    ensure_dir(os.path.join(cfg.out_dir, "meta"))
    cache = load_cache(cfg.data_dir)
    df = pd.read_parquet(os.path.join(cfg.data_dir, "dataset.parquet"))
    specs = load_json(os.path.join(cfg.out_dir, "calib", "specialists.json"))
    gen = load_json(os.path.join(cfg.out_dir, "calib", "general.json"))
    type_specs = [specs[t] for t in ["topic", "naive", "fake", "combined"]]

    print("=== META-DECISION EVALUATION (VAL split) ===")
    ev = evaluate_meta(type_specs, gen, df, cache, "val",
                       mode=cfg.meta_mode, cfg=cfg)
    ev["records"].to_csv(os.path.join(cfg.out_dir, "meta", "val_records.csv"),
                         index=False)

    print(f"mode={cfg.meta_mode}")
    print(f"  detection {ev['detection']:.4f} | ASR {ev['asr']:.4f} | "
          f"FPR {ev['fpr']:.4f} | F1 {ev['f1']:.4f}")
    print(f"  mean ASR across types "
          f"{ev['mean_asr'] if ev['mean_asr'] is None else round(ev['mean_asr'], 4)}")
    print(f"  attribution accuracy (detected injected) {ev['attr_accuracy']}")
    for t, v in ev["per_type"].items():
        print(f"    {t:10s} n={v['n']:3d} det={v['detection']:.4f} asr={v['asr']:.4f}")

    # alternative: global_max with a single threshold chosen on val
    evs = {sp["name"]: score_spec_on_split(sp, df, cache, "val", cfg.epsilon)
           for sp in type_specs}
    evg = score_spec_on_split(gen, df, cache, "val", cfg.epsilon)
    import numpy as np
    rows = df[df["split"] == "val"].reset_index(drop=True)
    maxv = np.zeros(len(rows))
    for i in range(len(rows)):
        vals = [ps["s"][i] for ps in evs.values()] + [evg["s"][i]]
        maxv[i] = max(vals)
    gt = choose_theta(maxv, rows["label"].to_numpy(), cfg.target_fpr)
    ev_global = evaluate_meta(type_specs, gen, df, cache, "val",
                              mode="global_max", cfg=cfg,
                              global_theta=gt["theta"])

    save_json({
        "mode": cfg.meta_mode,
        "global_max_theta": gt["theta"],
        "global_max_val": {k: ev_global[k] for k in
                           ("detection", "asr", "fpr", "f1", "mean_asr",
                            "attr_accuracy", "per_type")},
        "per_spec_val": {k: ev[k] for k in
                         ("detection", "asr", "fpr", "f1", "mean_asr",
                          "attr_accuracy", "per_type")},
    }, os.path.join(cfg.out_dir, "meta", "meta_config.json"))
    save_json({k: ev[k] for k in ev if k != "records"},
              os.path.join(cfg.out_dir, "meta", "val_eval.json"))

    print(f"\nglobal_max alternative (theta={gt['theta']:.4f} on val): "
          f"det={ev_global['detection']:.4f} asr={ev_global['asr']:.4f} "
          f"fpr={ev_global['fpr']:.4f}")
    ok = ev["fpr"] < cfg.target_fpr
    print(f"\nCriterion check: FPR < {cfg.target_fpr} -> "
          f"{'MET' if ok else 'NOT MET (' + str(round(ev['fpr'], 4)) + ')'}")
    print(f"Saved -> {os.path.join(cfg.out_dir, 'meta', 'meta_config.json')}")


if __name__ == "__main__":
    main()
