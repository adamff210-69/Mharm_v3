#!/usr/bin/env python3
"""13 — Robust attention calibration (no model).

The QuietRAG train-clean set has n=14 and at least one exploding R (~1e4)
when the pseudo-tail span is almost the whole short passage (body tokens ~ 0).
That wrecks z-scores, clean-p95 cutoffs, and any linear mix on raw R.

This script:
  1. Prints the exploding clean row(s) with span widths.
  2. Recalibrates attention on log1p(R) using train+val clean, after dropping
     rows whose passage-body is too short (W_p - W_i < min_body).
  3. Re-runs OR(hidden, attention) and a 2-feature logistic on (log1p(R), P).

Run after 12 (needs data/signals + out/calib/general.json):
  python 13_robust_attention.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from config import load_config
from multi_harm_common import signals as S
from multi_harm_common.io_utils import load_json, save_json
from multi_harm_common.metrics import auroc, tpr_fpr
from multi_harm_common.sigcache import load_cache

MIN_BODY = 8  # tokens of non-span passage required for a stable R


def _or_stats(att, hid, y):
    pred = (att | hid).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    return {
        "caught": tp / max(1, tp + fn),
        "false_alarm": fp / max(1, fp + tn),
        "att_only": int(np.sum(att & ~hid & (y == 1))),
        "hid_only": int(np.sum(hid & ~att & (y == 1))),
        "both": int(np.sum(att & hid & (y == 1))),
        "miss_both": int(np.sum(~att & ~hid & (y == 1))),
    }


def main():
    cfg = load_config()
    df = pd.read_parquet(os.path.join(cfg.data_dir, "dataset.parquet"))
    cache = load_cache(cfg.data_dir)
    gen = load_json(os.path.join(cfg.out_dir, "calib", "general.json"))
    head = tuple(gen["head"] if gen.get("head") else gen["head_shared"])

    ids = df["id"].tolist()
    sub = cache.subset(ids)
    R = np.array([S.head_ratio(sub["masses"][i][head], cfg.epsilon, sub["widths"][i])
                  for i in range(len(ids))])
    P = np.array([S.probe_probs(sub["hid"][gen["L_star"]][i], gen["probe"])
                  for i in range(len(ids))])
    Wp = np.array([sub["widths"][i][0] for i in range(len(ids))])
    Wi = np.array([sub["widths"][i][1] for i in range(len(ids))])
    body = np.maximum(0, Wp - Wi)
    y = sub["labels"].astype(int)
    split = df.set_index("id").loc[ids, "split"].to_numpy()
    types = np.array(sub["types"])
    plen = df.set_index("id").loc[ids, "passage"].map(len).to_numpy()

    tab = pd.DataFrame({
        "id": ids, "split": split, "attack_type": types, "label": y,
        "R": R, "logR": np.log1p(np.clip(R, 0, None)), "P": P,
        "W_p": Wp, "W_i": Wi, "body": body, "passage_chars": plen,
    })
    zp = (tab["P"] - gen["p_mu"]) / max(gen["p_sd"], 1e-12)
    tab["hid_fire"] = zp.to_numpy() > gen["theta"]

    print("=== exploding clean R ===")
    clean = tab[tab.label == 0].sort_values("R", ascending=False)
    print(clean[["id", "split", "R", "logR", "P", "W_p", "W_i", "body",
                 "passage_chars"]].head(8).to_string(index=False))
    print("\nclean R quantiles:\n", clean.R.quantile([0.5, 0.8, 0.9, 0.95, 1.0]).to_string())
    print("injected R quantiles:\n", tab[tab.label == 1].R.quantile([0.5, 0.8, 0.9, 0.95, 1.0]).to_string())
    print(f"\nclean rows with body < {MIN_BODY} tokens: "
          f"{int(((tab.label == 0) & (tab.body < MIN_BODY)).sum())} / {(tab.label == 0).sum()}")

    stable = tab.body >= MIN_BODY
    print(f"dropped {int((~stable).sum())} rows with body < {MIN_BODY} for attention cutoff")

    # train+val clean, stable only
    ref = tab[stable & (tab.label == 0) & tab.split.isin(["train", "val"])]
    print(f"\nrobust clean reference n={len(ref)} (train+val, body>={MIN_BODY})")
    if len(ref) < 4:
        print("too few stable clean rows — abort")
        sys.exit(1)
    print("  raw R   median={:.3f} p95={:.3f} max={:.3f}".format(
        ref.R.median(), ref.R.quantile(0.95), ref.R.max()))
    print("  log1p R median={:.3f} p95={:.3f} max={:.3f}".format(
        ref.logR.median(), ref.logR.quantile(0.95), ref.logR.max()))

    th_R = float(ref.R.quantile(0.95))
    th_log = float(ref.logR.quantile(0.95))
    print(f"\ncutoffs from this reference: R p95={th_R:.4f}  log1p(R) p95={th_log:.4f}")

    print("\n=== attention-only (robust cutoff) ===")
    for name, th, col in (("raw R p95", th_R, "R"), ("log1p(R) p95", th_log, "logR")):
        print(f"\n  {name} = {th:.4f}")
        for sp in ("train", "val", "test"):
            sl = tab[tab.split == sp]
            tpr, fpr = tpr_fpr(sl.label.to_numpy(), sl[col].to_numpy(), th)
            print(f"    {sp:5s} caught={tpr:.3f}  false-alarm={fpr:.3f}  "
                  f"n_inj={int(sl.label.sum())} n_clean={int((sl.label == 0).sum())}")

    print("\n=== hidden misses vs robust attention ===")
    hid_miss = tab[(tab.label == 1) & (~tab.hid_fire)]
    print(hid_miss[["id", "split", "attack_type", "R", "logR", "P", "body"]].to_string(index=False))
    n_res_R = int((hid_miss.R > th_R).sum())
    n_res_log = int((hid_miss.logR > th_log).sum())
    print(f"\n  of {len(hid_miss)} hidden misses, raw-R p95 would catch {n_res_R}, "
          f"log1p p95 would catch {n_res_log}")

    print("\n=== OR-rule on TEST (hidden current OR robust attention) ===")
    te = tab[tab.split == "test"]
    yte = te.label.to_numpy()
    hid = te.hid_fire.to_numpy()
    for tag, att in (("raw R p95", te.R.to_numpy() > th_R),
                     ("log1p p95", te.logR.to_numpy() > th_log)):
        st = _or_stats(att, hid, yte)
        hid_st = _or_stats(np.zeros_like(hid), hid, yte)
        print(f"  hidden only:          caught={hid_st['caught']:.3f}  fa={hid_st['false_alarm']:.3f}")
        print(f"  OR {tag:16s} caught={st['caught']:.3f}  fa={st['false_alarm']:.3f}  "
              f"att-only extra={st['att_only']}  still missed={st['miss_both']}")

    print("\n=== logistic (train -> test) ===")
    tr = tab[tab.split == "train"]

    def fit_eval(cols, tag):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(tr[cols].to_numpy(), tr.label.to_numpy())
        proba = clf.predict_proba(te[cols].to_numpy())[:, 1]
        au = auroc(yte, proba)
        au_h = auroc(yte, te.P)
        au_a = auroc(yte, te[cols[0]] if cols[0] != "P" else te.R)
        print(f"  {tag:20s} coef={np.round(clf.coef_[0], 3).tolist()}  "
              f"intercept={clf.intercept_[0]:+.3f}  test AUROC={au:.4f}  "
              f"(hidden={au_h:.4f})")
        return au, au_h, {c: float(w) for c, w in zip(cols, clf.coef_[0])}

    a_raw, a_h, w_raw = fit_eval(["R", "P"], "raw R + P")
    a_log, _, w_log = fit_eval(["logR", "P"], "log1p(R) + P")
    gain = a_log - a_h
    print(f"\n  log1p mix minus hidden-only AUROC: {gain:+.4f}")
    if gain > 1e-4:
        print("  -> mix BEATS hidden on ranking. Small, but this is the fusion opening.")
    else:
        print("  -> mix does NOT beat hidden on overall ranking (expected if hidden "
              "already ~0.99). Check OR-rule extra catches above — that is the operating-point question.")

    out = {
        "min_body_tokens": MIN_BODY,
        "n_clean_exploding": int(((tab.label == 0) & (tab.R > 100)).sum()),
        "top_clean_R": clean.head(5)[["id", "split", "R", "W_p", "W_i", "body",
                                      "passage_chars"]].to_dict("records"),
        "ref_n": int(len(ref)),
        "th_R_p95": th_R,
        "th_logR_p95": th_log,
        "hidden_miss_ids": hid_miss.id.tolist(),
        "hidden_miss_caught_by_R": n_res_R,
        "hidden_miss_caught_by_logR": n_res_log,
        "logistic_raw": w_raw,
        "logistic_log": w_log,
        "auroc_log_mix_test": a_log,
        "auroc_hidden_test": a_h,
        "auroc_gain": gain,
        "note": ("n_hidden_miss is small (QuietRAG). A gain of one test catch is "
                 "real but not a fusion paper by itself."),
    }
    save_json(out, os.path.join(cfg.out_dir, "experiments", "robust_attention.json"))
    tab.to_csv(os.path.join(cfg.out_dir, "experiments", "per_example_scores_robust.csv"),
               index=False)
    print("\nWrote out/experiments/robust_attention.json")


if __name__ == "__main__":
    main()
