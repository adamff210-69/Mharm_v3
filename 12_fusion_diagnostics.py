#!/usr/bin/env python3
"""12 — Fusion room check (no model). Uses cached QuietRAG/MS-MARCO signals.

Answers, in order:
  1. Do attention and hidden-state *disagree* on any real examples?
     If they almost always agree, no mixer can beat hidden-only on this data.
  2. Recalibrate the attention cutoff from *clean documents only*
     (95th / 99th percentile of raw R), vs the broken z-score theta (~5).

Does NOT retrain fusion. A 2-feature logistic is reported only as a probe of
whether a learned mix could help on the disagreement subset.

Run (after 03+06):  python 12_fusion_diagnostics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import load_config
from multi_harm_common import signals as S
from multi_harm_common.io_utils import load_json, save_json
from multi_harm_common.metrics import auroc, pearson, tpr_fpr
from multi_harm_common.sigcache import load_cache


def _or_rule(att_flag, hid_flag, y):
    pred = (att_flag | hid_flag).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return {
        "caught": tp / max(1, tp + fn),
        "missed": fn / max(1, tp + fn),
        "false_alarm": fp / max(1, fp + tn),
        "n_caught_by_att_only": int(np.sum((att_flag == 1) & (hid_flag == 0) & (y == 1))),
        "n_caught_by_hid_only": int(np.sum((hid_flag == 1) & (att_flag == 0) & (y == 1))),
        "n_caught_by_both": int(np.sum((att_flag == 1) & (hid_flag == 1) & (y == 1))),
        "n_missed_both": int(np.sum((att_flag == 0) & (hid_flag == 0) & (y == 1))),
    }


def main():
    cfg = load_config()
    df = pd.read_parquet(os.path.join(cfg.data_dir, "dataset.parquet"))
    cache = load_cache(cfg.data_dir)
    gen = load_json(os.path.join(cfg.out_dir, "calib", "general.json"))
    if not gen:
        print("Run 06_calibrate_general.py first.")
        sys.exit(1)

    head = tuple(gen["head"] if gen.get("head") else gen["head_shared"])
    print(f"Shared detector: L*={gen['L_star']} head={head} alpha={gen['alpha']:.2f} "
          f"theta={gen['theta']:.4f}")

    # raw R + P on every cached id (aligned with dataset)
    ids = df["id"].tolist()
    sub = cache.subset(ids)
    R = np.array([S.head_ratio(sub["masses"][i][head], cfg.epsilon, sub["widths"][i])
                  for i in range(len(ids))])
    P = np.array([S.probe_probs(sub["hid"][gen["L_star"]][i], gen["probe"])
                  for i in range(len(ids))])
    y = sub["labels"].astype(int)
    types = np.array(sub["types"])
    split = df.set_index("id").loc[ids, "split"].to_numpy()

    zr = (R - gen["r_mu"]) / max(gen["r_sd"], 1e-12)
    zp = (P - gen["p_mu"]) / max(gen["p_sd"], 1e-12)

    tab = pd.DataFrame({
        "id": ids, "split": split, "attack_type": types, "label": y,
        "R": R, "P": P, "zr": zr, "zp": zp,
    })
    tab.to_csv(os.path.join(cfg.out_dir, "experiments", "per_example_scores.csv"),
               index=False)

    print("\n=== 1. Do the two clues disagree? ===")
    print(f"  corr(R, P) all={pearson(R, P):+.3f}  injected={pearson(R[y==1], P[y==1]):+.3f}  "
          f"clean={pearson(R[y==0], P[y==0]):+.3f}")

    # hidden miss = P below the operating cutoff implied by general theta on zp
    hid_flag = zp > gen["theta"]
    # also rank-based: hidden's lowest injected vs highest clean
    room = {}
    for sp_name, msplit in (("all", np.ones(len(y), bool)),
                            ("test", split == "test"),
                            ("val", split == "val")):
        m = msplit
        sl = tab.loc[m].reset_index(drop=True)
        yy = sl["label"].to_numpy()
        pp = sl["P"].to_numpy()
        rr = sl["R"].to_numpy()
        zpp = sl["zp"].to_numpy()
        hid_pred = (zpp > gen["theta"]).astype(int)
        hid_miss = (hid_pred == 0) & (yy == 1)
        hid_fa = (hid_pred == 1) & (yy == 0)
        n_miss = int(hid_miss.sum())
        n_fa = int(hid_fa.sum())
        miss_R = rr[hid_miss]
        clean_R = rr[yy == 0]
        att_would = np.array([])
        if n_miss and len(clean_R):
            thr95 = float(np.quantile(clean_R, 0.95)) if len(clean_R) >= 2 else float(clean_R.max())
            att_would = miss_R > thr95
        print(f"\n  [{sp_name}] n={int(m.sum())} inj={int(yy.sum())} clean={int((yy==0).sum())}")
        print(f"    hidden (current theta={gen['theta']:.3f}): "
              f"miss {n_miss}/{int(yy.sum())}  false-alarm {n_fa}/{int((yy==0).sum())}")
        print(f"    ranking AUROC  attention={auroc(yy, rr):.4f}  hidden={auroc(yy, pp):.4f}")
        if n_miss:
            print(f"    hidden misses: ids={sl.loc[hid_miss, 'id'].tolist()}")
            print(f"      their R={np.round(miss_R, 3).tolist()}  P={np.round(pp[hid_miss], 4).tolist()}")
            if len(clean_R):
                print(f"      clean R  median={np.median(clean_R):.3f}  p95={np.quantile(clean_R, 0.95):.3f}")
                n_rescue = int(np.sum(att_would)) if len(att_would) else 0
                print(f"      attention (clean p95) would catch {n_rescue}/{n_miss} of those misses")
        else:
            print("    hidden missed 0 injected on this split — fusion has nothing to rescue")
        room[sp_name] = {
            "n": int(m.sum()), "n_inj": int(yy.sum()), "n_clean": int((yy == 0).sum()),
            "hidden_miss": n_miss, "hidden_false_alarm": n_fa,
            "auroc_attention": auroc(yy, rr), "auroc_hidden": auroc(yy, pp),
            "corr_R_P": pearson(rr, pp),
        }

    print("\n=== 2. Attention cutoff from CLEAN documents only ===")
    train = tab[tab["split"] == "train"]
    clean_R_tr = train.loc[train["label"] == 0, "R"].to_numpy()
    if len(clean_R_tr) < 2:
        print("  not enough train clean rows")
        clean_stats = {}
    else:
        print(f"  train clean n={len(clean_R_tr)}  R median={np.median(clean_R_tr):.3f} "
              f"max={clean_R_tr.max():.3f}")
        clean_stats = {}
        for name, q in (("p95", 0.95), ("p99", 0.99)):
            th = float(np.quantile(clean_R_tr, q))
            print(f"\n  cutoff = {name} of train-clean R  ({th:.4f})")
            for sp in ("train", "val", "test"):
                sl = tab[tab["split"] == sp]
                tpr, fpr = tpr_fpr(sl["label"].to_numpy(), sl["R"].to_numpy(), th)
                print(f"    {sp:5s}  caught={tpr:.3f}  false-alarm={fpr:.3f}  "
                      f"(n_inj={int(sl.label.sum())} n_clean={int((sl.label==0).sum())})")
            # vs published z-score theta on zr
            tpr_old, fpr_old = tpr_fpr(tab.loc[tab.split == "test", "label"],
                                       tab.loc[tab.split == "test", "zr"], gen["theta"])
            clean_stats[name] = {"theta_R": th}
        print(f"\n  published attention z-threshold theta={gen['theta']:.4f} on TEST: "
              f"caught={tpr_old:.3f} false-alarm={fpr_old:.3f}  "
              f"(this is why row-1 ASR was 1.0)")

        # OR of (clean-p95 attention) and (hidden current)
        th95 = float(np.quantile(clean_R_tr, 0.95))
        print("\n  OR-rule on TEST: fire if hidden fires OR attention R > train-clean p95")
        te = tab[tab["split"] == "test"]
        att_f = te["R"].to_numpy() > th95
        hid_f = te["zp"].to_numpy() > gen["theta"]
        yy = te["label"].to_numpy()
        orr = _or_rule(att_f, hid_f, yy)
        hid_only = _or_rule(np.zeros_like(hid_f), hid_f, yy)
        print(f"    hidden only:  caught={hid_only['caught']:.3f}  false-alarm={hid_only['false_alarm']:.3f}")
        print(f"    OR with att:  caught={orr['caught']:.3f}  false-alarm={orr['false_alarm']:.3f}")
        print(f"    extra attacks caught by attention only: {orr['n_caught_by_att_only']}")
        print(f"    extra attacks caught by hidden only:    {orr['n_caught_by_hid_only']}")
        print(f"    still missed by both:                   {orr['n_missed_both']}")
        room["or_rule_test"] = orr
        room["hidden_only_test"] = hid_only
        room["att_clean_p95"] = th95

    # tiny logistic on (R, P) — only to see if a learned mix finds the misses
    print("\n=== 3. Preview: 2-feature logistic on (R, P), train -> test ===")
    from sklearn.linear_model import LogisticRegression
    tr = tab[tab["split"] == "train"]
    te = tab[tab["split"] == "test"]
    Xtr = tr[["R", "P"]].to_numpy()
    Xte = te[["R", "P"]].to_numpy()
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, tr["label"].to_numpy())
    proba = clf.predict_proba(Xte)[:, 1]
    print(f"  coef R={clf.coef_[0,0]:+.3f}  P={clf.coef_[0,1]:+.3f}  intercept={clf.intercept_[0]:+.3f}")
    print(f"  test AUROC logistic={auroc(te['label'], proba):.4f}  "
          f"hidden-P={auroc(te['label'], te['P']):.4f}  attention-R={auroc(te['label'], te['R']):.4f}")
    print("  (if logistic ≈ hidden, a learned mix also has nothing to add on this set)")

    out = {
        "head": list(head), "L_star": gen["L_star"], "alpha": gen["alpha"],
        "published_theta": gen["theta"],
        "corr": {"all": pearson(R, P),
                 "injected": pearson(R[y == 1], P[y == 1]),
                 "clean": pearson(R[y == 0], P[y == 0])},
        "splits": room,
        "logistic": {"coef_R": float(clf.coef_[0, 0]),
                     "coef_P": float(clf.coef_[0, 1]),
                     "auroc_test": auroc(te["label"], proba),
                     "auroc_hidden_test": auroc(te["label"], te["P"]),
                     "auroc_att_test": auroc(te["label"], te["R"])},
        "clean_percentile": clean_stats,
        "verdict": (
            "NO ROOM for fusion on this data"
            if room.get("test", {}).get("hidden_miss", 1) == 0
            else "SOME hidden misses — check whether attention ranks them above clean p95"
        ),
    }
    save_json(out, os.path.join(cfg.out_dir, "experiments", "fusion_diagnostics.json"))
    print(f"\nWrote out/experiments/fusion_diagnostics.json")
    print(f"Wrote out/experiments/per_example_scores.csv")
    print(f"\nVERDICT: {out['verdict']}")


if __name__ == "__main__":
    main()
