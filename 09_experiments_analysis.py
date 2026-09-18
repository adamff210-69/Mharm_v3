#!/usr/bin/env python3
"""09 — Main experiments, ablations and analysis (v3 Phase 4 + 5).

Produces every table of the paper, evaluated on the TEST split (val was used
for the one meta-level threshold; everything else is calibration-only):

  Table A   main per-attack-type performance of Multi-HARM (meta, per_spec)
  Table B   baselines: 4.8-row1 attention-shared, PIShield-style hidden-only,
            HARM_general (4.8-row2), per-specialist-alone, Multi-HARM
  Table C   ablations: signal halves; top-1 vs top-K head ensemble;
            per_spec vs global_max meta
  Table D   attack-type attribution: accuracy + confusion (v3 Phase 3/§4)
  Table E   unseen-attack generalization (v3 §4.5), held-out type = cfg.unseen_type
  §4.3      signal independence: pairwise Pearson of specialist scores (val)
  §4.4      cross-specialist generalization: off-diagonal firing matrix (val)
  §4.8      the 3-row spine table (assembled from 05/06 + row 3 computed here)
  latency   shared-forward-pass overhead measurement (v3 shared-computation
            claim; --with-model adds the full forward comparison)

Writes out/experiments/*.csv + .json and out/experiments/SUMMARY.md
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from config import load_config, ATTACK_TYPES
from multi_harm_common import signals as S
from multi_harm_common.calibrate import (calibrate_specialist,
                                         recalibrate_general_without,
                                         select_calib_ids)
from multi_harm_common.detect import (evaluate_meta, meta_decision,
                                      per_type_auroc, score_map,
                                      score_spec_on_split, type_vs_clean_ids)
from multi_harm_common.io_utils import load_json, save_json
from multi_harm_common.metrics import auroc, pearson, tpr_fpr
from multi_harm_common.sigcache import load_cache, usable_df

TYPES = ATTACK_TYPES
EXP = lambda cfg: os.path.join(cfg.out_dir, "experiments")


def load_all(cfg):
    cache = load_cache(cfg.data_dir)
    df = pd.read_parquet(os.path.join(cfg.data_dir, "dataset.parquet"))
    df = usable_df(df, cache)          # drop anything 03 could not extract
    hstar = load_json(os.path.join(cfg.out_dir, "calib", "H_star.json"))
    specs = load_json(os.path.join(cfg.out_dir, "calib", "specialists.json"))
    gen = load_json(os.path.join(cfg.out_dir, "calib", "general.json"))
    meta_cfg = load_json(os.path.join(cfg.out_dir, "meta", "meta_config.json"),
                         default={})
    return cache, df, hstar, specs, gen, meta_cfg


# (v3.0 had a local per_type_auroc_of here that filtered to the attack type
# only — an all-positive subset, so every AUROC came out at exactly 0.5. The
# definition now lives in detect.per_type_auroc and includes the clean rows.)
per_type_auroc_of = per_type_auroc


# ---------------------------------------------------------------------------

def table_a(cfg, cache, df, specs, gen, meta_cfg):
    ev = evaluate_meta([specs[t] for t in TYPES], gen, df, cache, "test",
                       mode=cfg.meta_mode, cfg=cfg,
                       global_theta=meta_cfg.get("global_max_theta"))
    ev["records"].to_csv(os.path.join(EXP(cfg), "test_records.csv"), index=False)
    rows = []
    for t in TYPES + ["ALL"]:
        if t == "ALL":
            rows.append({"config": "Multi-HARM", "type": "all",
                         "n": ev["n"], "detection": ev["detection"],
                         "asr": ev["asr"], "fpr": ev["fpr"], "f1": ev["f1"]})
        else:
            v = ev["per_type"].get(t)
            if v:
                rows.append({"config": "Multi-HARM", "type": t, "n": v["n"],
                             "detection": v["detection"], "asr": v["asr"],
                             "fpr": np.nan, "f1": np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(EXP(cfg), "tableA_main.csv"), index=False)
    return ev


def table_b(cfg, cache, df, specs, gen, row1, row2, hid_base):
    """Per-config: per-type AUROC + overall ASR/FPR (threshold configs where
    applicable)."""
    rows = []
    for name, pt in [("attn-shared (4.8 r1)", row1["per_type_auroc"]),
                     ("fused-shared HARM_general (4.8 r2)",
                      row2["per_type_auroc"]),
                     ("hidden-only shared (PIShield-style)",
                      hid_base["per_type_auroc"])]:
        for t in TYPES:
            rows.append({"config": name, "type": t,
                         "auroc": pt.get(t, np.nan)})
    rows.append({"config": "hidden-only shared (PIShield-style)",
                 "type": "all", "auroc": hid_base["auroc_overall"]})
    rows.append({"config": "attn-shared (4.8 r1)", "type": "all",
                 "auroc": row1["auroc_overall"]})

    # threshold-based configs: per-specialist alone (own theta, own type)
    for t in TYPES:
        sp = specs[t]
        ev = score_spec_on_split(sp, df, cache, "test", cfg.epsilon)
        m = ev["types"] == t            # injected-only: correct for det/ASR,
        tpr, fpr = tpr_fpr(ev["labels"][m], ev["s"][m], sp["theta"])
        au = per_type_auroc(score_map(ev), cache, df, "test", [t])[t]  # ...
        rows.append({"config": f"specialist-alone {t}", "type": t,
                     "auroc": au,      # ...not for the AUROC, which is scored
                     "detection": float(tpr), "asr": float(1 - tpr),
                     "n_inj": int(m.sum())})   # against clean below
    return pd.DataFrame(rows), None


def table_c(cfg, cache, df, specs, gen, meta_cfg):
    rows = []
    # (i) halves on test (calib z-stats, no test-set fitting). Each cell is
    # AUROC over {injected of this type} + {clean}, per detect.type_vs_clean_ids.
    for t in TYPES:
        sp = specs[t]
        ev = score_spec_on_split(sp, df, cache, "test", cfg.epsilon)
        for tag, key in (("attention-half only", "zr"),
                         ("hidden-half only", "zp"),
                         ("fused (as calibrated)", "s")):
            rows.append({"ablation": tag, "type": t,
                         "auroc": per_type_auroc(score_map(ev, key), cache, df,
                                                 "test", [t])[t]})
    # (ii) top-1 vs top-K head ensemble (re-calibrated on calib set). z-stats
    # and theta come from the calibration set; the test-side score covers the
    # WHOLE test split so the per-type AUROC has both classes.
    mt = cache.subset(df[df["split"] == "test"]["id"].tolist())
    for t in TYPES:
        sp = specs[t]
        sub = cache.subset(sp["calib"]["ids"])
        nc, nt = len(sub["ids"]), len(mt["ids"])

        def zp_of(src, n):
            return ((np.array([S.probe_probs(src["hid"][sp["L_star"]][i],
                                              sp["probe"]) for i in range(n)])
                      - sp["p_mu"]) / max(sp["p_sd"], 1e-12))
        zp_c, zp_t = zp_of(sub, nc), zp_of(mt, nt)

        headsk = [tuple(h) for h, _ in sp["top_k"]]
        headsets = {"top-1 head": [tuple(sp["head"])]}
        if len(headsk) > 1:
            headsets[f"top-{len(headsk)} heads"] = headsk
        for tag, heads in headsets.items():
            def r_of(src, n, heads=heads):
                return np.array([np.mean([S.head_ratio(src["masses"][i][h],
                                                        cfg.epsilon,
                                                        src["widths"][i])
                                          for h in heads])
                                 for i in range(n)])
            zr_c, mu, sd = S.zscore(r_of(sub, nc))
            fused_c = sp["alpha"] * zr_c + (1 - sp["alpha"]) * zp_c
            theta = S.choose_theta(fused_c, sub["labels"], sp["fpr_budget"])["theta"]
            s_t = (sp["alpha"] * ((r_of(mt, nt) - mu) / max(sd, 1e-12))
                   + (1 - sp["alpha"]) * zp_t)
            inj = mt["types"] == t
            tpr, _ = tpr_fpr(mt["labels"][inj], s_t[inj], theta)
            rows.append({"ablation": tag, "type": t,
                         "auroc": per_type_auroc(dict(zip(mt["ids"], s_t)),
                                                  cache, df, "test", [t])[t],
                         "asr": float(1 - tpr), "n_inj": int(inj.sum())})
    # (iii) meta modes
    ev1 = evaluate_meta([specs[t] for t in TYPES], gen, df, cache, "test",
                        mode="per_spec", cfg=cfg)
    ev2 = evaluate_meta([specs[t] for t in TYPES], gen, df, cache, "test",
                        mode="global_max", cfg=cfg,
                        global_theta=meta_cfg.get("global_max_theta"))
    for nm, ev in (("meta per_spec (default)", ev1),
                   ("meta global_max", ev2)):
        rows.append({"ablation": nm, "type": "all",
                     "detection": ev["detection"], "asr": ev["asr"],
                     "fpr": ev["fpr"]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(EXP(cfg), "tableC_ablations.csv"), index=False)
    return out


def table_e_unseen(cfg, cache, df, specs, gen, hstar):
    u = cfg.unseen_type
    gen_u = recalibrate_general_without(df, cache, cfg, hstar, u)
    seen = [t for t in TYPES if t != u]
    ev = evaluate_meta([specs[t] for t in seen], gen_u, df, cache, "test",
                       mode="per_spec", cfg=cfg)
    v_u = ev["per_type"].get(u, {})
    v_seen = [ev["per_type"][t]["asr"] for t in seen if t in ev["per_type"]]
    mean_seen = float(np.mean(v_seen)) if v_seen else None
    res = {
        "held_out_type": u,
        "asr_unseen": v_u.get("asr"), "detection_unseen": v_u.get("detection"),
        "mean_asr_seen": mean_seen,
        "gap_points": (v_u.get("asr", 0) - mean_seen) if mean_seen is not None else None,
        "within_10_points": (abs(v_u.get("asr", 1) - mean_seen) <= 0.10
                             if mean_seen is not None else None),
        "general_without_recalib": {k: gen_u[k]
                                    for k in ("L_star", "alpha", "theta", "auroc")},
    }
    save_json(res, os.path.join(EXP(cfg), "tableE_unseen.json"))
    return res


def sec_43_independence(cfg, cache, df, specs):
    evs = {t: score_spec_on_split(specs[t], df, cache, "val", cfg.epsilon)
           for t in TYPES}
    n = len(evs[TYPES[0]]["ids"])
    mat = pd.DataFrame(index=TYPES, columns=TYPES, dtype=float)
    for a in TYPES:
        for b in TYPES:
            mat.loc[a, b] = pearson(evs[a]["s"][:n], evs[b]["s"][:n]) \
                if a != b else 1.0
    mat.to_csv(os.path.join(EXP(cfg), "sec43_pearson_val.csv"))
    pairs = {f"{a}|{b}": float(mat.loc[a, b])
             for a in TYPES for b in TYPES if a < b}
    save_json({"pairs": pairs,
               "criterion_max_lt": bool(max(pairs.values()) < 0.5),
               "split": "val", "n": int(n)},
              os.path.join(EXP(cfg), "sec43_pearson_val.json"))
    return pairs, bool(max(pairs.values()) < 0.5)


def sec_44_cross(cfg, cache, df, specs):
    """For each TRUE type (val, injected), which specialist's fused score is
    highest? Off-diagonals = cross-specialist firing."""
    evs = {t: score_spec_on_split(specs[t], df, cache, "val", cfg.epsilon)
           for t in TYPES}
    m = df[(df["split"] == "val") & (df["label"] == 1)]
    rows = m.to_dict("records")
    by_id = {t: {sid: s for sid, s in zip(evs[t]["ids"], evs[t]["s"])}
             for t in TYPES}
    counts = {t: {u: 0 for u in TYPES} for t in TYPES}
    for r in rows:
        tt = r["attack_type"]
        if tt not in counts:
            continue
        scores = {u: by_id[u].get(r["id"], -1e9) for u in TYPES}
        top = max(scores, key=scores.get)
        counts[tt][top] += 1
    mat = pd.DataFrame(counts)
    mat.to_csv(os.path.join(EXP(cfg), "sec44_cross_val.csv"))
    tot = int(mat.values.sum())
    diag = int(sum(mat.loc[t, t] for t in TYPES if t in mat.index and t in mat.columns))
    save_json({"matrix": {t: {u: int(mat.loc[t, u]) for u in TYPES}
                          for t in TYPES if t in mat.index},
               "n": tot, "own_type_fraction": (diag / tot if tot else None),
               "note": ("rows = true attack type, cols = which specialist scored "
                        "it highest (VAL, injected). This is the honest "
                        "attribution number: Table D's argmax accuracy is biased "
                        "upward because every specialist is calibrated on its "
                        "own type.")},
              os.path.join(EXP(cfg), "sec44_cross_val.json"))
    return mat


def width_diagnostics(cfg, cache, df, hstar):
    """Span-width audit (Table A robustness, v3 review fix).

    Reports per-type injection-span token counts on test and the Pearson
    correlation between the pooled-head attention ratio R and span width W_i
    (overall and within injected). R is the span-width-INVARIANT intensity
    ratio against the passage BODY, so it should be (near-)uncorrelated with
    width; any residual correlation is reported so the paper can state it
    honestly.
    """
    m = cache.subset(df[df["split"] == "test"]["id"].tolist())
    head = tuple(hstar["best_head"])
    r = np.array([S.head_ratio(m["masses"][i][head], cfg.epsilon, m["widths"][i])
                  for i in range(len(m["ids"]))])
    w_i = np.array([w[1] for w in m["widths"]], dtype=float)
    w_p = np.array([w[0] for w in m["widths"]], dtype=float)
    per_type = {}
    for t in TYPES + ["clean"]:
        sel = m["types"] == t
        if sel.any():
            per_type[t] = {"n": int(sel.sum()),
                           "w_i_mean": float(w_i[sel].mean()),
                           "w_i_median": float(np.median(w_i[sel])),
                           "w_i_std": float(w_i[sel].std()),
                           "w_p_mean": float(w_p[sel].mean())}
    inj = m["labels"] == 1
    res = {
        "note": ("R is the span-width-INVARIANT per-token intensity ratio "
                 "(m_qi/W_i)/((m_qp-m_qi)/(W_p-W_i)) — i.e. normalized against "
                 "the passage BODY, not the whole passage; these width stats "
                 "show payload lengths per type and any residual R-width "
                 "association."),
        "per_type_width_test": per_type,
        "r_width_corr_all_test": pearson(r, w_i),
        "r_width_corr_injected_test": (pearson(r[inj], w_i[inj])
                                       if inj.sum() > 2 else None),
    }
    # The width-invariance claim has to be measured against the estimator it
    # replaced, not asserted. v3.0's raw sum-ratio m_qi_sum / m_qp_sum is not
    # width-normalized, so where payload length varies by attack type it tracks
    # that variation; the deployed per-column-mean ratio must not. Both
    # correlations are reported so the robustness claim is falsifiable.
    naive = []
    for j in range(len(m["ids"])):
        mm = np.asarray(m["masses"][j][head], dtype=np.float64)   # (m_qp, m_qi, m_qq)
        naive.append((mm[1] + cfg.epsilon) / (mm[0] - mm[1] + cfg.epsilon))
    naive = np.array(naive)
    res["naive_sumratio_width_corr_injected_test"] = (
        pearson(naive[inj], w_i[inj]) if inj.sum() > 2 else None)
    res["interpretation"] = (
        "corr(head_ratio, W_i) near zero while corr(sum-ratio, W_i) is "
        "materially nonzero is the width confound v3.1 removes; if both are flat "
        "this dataset's width range is too narrow to demonstrate it either way")
    save_json(res, os.path.join(EXP(cfg), "span_width.json"))
    save_json(res, os.path.join(cfg.out_dir, "validation",
                                "width_invariance.json"))
    return res


def sec_48_table(cfg, cache, df, specs, gen, row1, row2):
    """Row 3: fused, per-attack specialized. Per-type AUROC: score each test
    sample with its OWN type's specialist (that is what the meta layer does
    per type)."""
    per_type = {}
    for t in TYPES:
        sp = specs[t]
        ev = score_spec_on_split(sp, df, cache, "test", cfg.epsilon)
        # each test sample is scored by its OWN type's specialist; the AUROC is
        # measured against the clean rows of that split (two classes — see
        # detect.type_vs_clean_ids)
        per_type[t] = per_type_auroc(score_map(ev), cache, df, "test", [t])[t]
    vals = [v for v in per_type.values() if v is not None]
    table = {
        "attention-only shared (4.8 r1)": dict(row1["per_type_auroc"],
                                               spread=row1["spread"]),
        "fused shared HARM_general (4.8 r2)": dict(row2["per_type_auroc"],
                                                   spread=row2["spread"]),
        "fused per-attack specialized (row 3)": dict(per_type,
                                                     spread=(max(vals) - min(vals)
                                                             if vals else float("nan"))),
    }
    save_json({"rows": table,
               "eval_set": ("per type: injected-of-type + all clean rows of the "
                            "test split (two-class AUROC; v3.0 filtered to the "
                            "type only, which returned 0.5 everywhere)")},
              os.path.join(EXP(cfg), "table_48.json"))
    return table


def calib_size_sweep(cfg, cache, df, hstar, sizes=(40, 80, 160)):
    """How big must the per-specialist calibration set be?

    The paper's central claim is that per-attack specialization helps, and the
    whole calibration budget is one constant (cfg.calib_per_specialist=160). If
    the specialized-vs-shared advantage only exists at large n, that is a
    different (and honest) finding than a gain that survives at n=40. Cheap:
    re-calibration runs entirely on cached signals.
    """
    import dataclasses
    rows = []
    n_heads = max(h for (_, h) in next(iter(cache.masses.values()))) + 1
    n_layers_cached = max(l for (l, _) in next(iter(cache.masses.values()))) + 1
    for size in sizes:
        c2 = dataclasses.replace(cfg, calib_per_specialist=int(size),
                                 calib_h_samples=int(size))
        for t in TYPES:
            sp = calibrate_specialist(t, df, cache, c2, hstar,
                                      use_shared_head=False, seed=cfg.seed + 7)
            ev = score_spec_on_split(sp, df, cache, "test", cfg.epsilon)
            # a head outside the cached grid would score every sample with the
            # same epsilon-floored value and still hand back an AUROC
            if not (0 <= sp["head"][0] < n_layers_cached
                    and 0 <= sp["head"][1] < n_heads):
                raise RuntimeError(
                    f"specialist '{t}' selected head {sp['head']} outside the "
                    f"cached {n_layers_cached}x{n_heads} grid")
            rows.append({
                "calib_n": int(size), "type": t,
                "L_star": sp["L_star"], "head": f"{sp['head'][0]}x{sp['head'][1]}",
                "alpha": sp["alpha"],
                "att_half_calib": sp["auroc"]["att_head"],
                "hid_half_calib": sp["auroc"]["hid"],
                "fused_calib": sp["auroc"]["fused"],
                "fused_heldout": sp["auroc"]["fused_eval"],
                "fused_test": per_type_auroc(score_map(ev), cache, df, "test",
                                             [t])[t],
            })
        print(f"  calib_n={size}: " + " ".join(
            f"{r['type']}=" + (f"{r['fused_test']:.4f}"
                               if r["fused_test"] is not None else "n/a")
            for r in rows[-len(TYPES):]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(EXP(cfg), "calib_size_sweep.csv"), index=False)
    piv = (out.groupby("calib_n")[["fused_calib", "fused_heldout", "fused_test"]]
              .mean())
    piv.to_csv(os.path.join(EXP(cfg), "calib_size_sweep_summary.csv"))
    save_json(piv.reset_index().to_dict("records"),
              os.path.join(EXP(cfg), "calib_size_sweep.json"))
    return out


def latency_measure(cfg, cache, df, specs, gen, with_model=False):
    t0 = df[df["split"] == "test"].sample(5, random_state=0).to_dict("records")
    import multi_harm_common.detect as D
    # scoring-only overhead (numpy): 5 specialists vs 1 specialist
    s0 = t0[0]
    enc_masses = cache.masses[s0["id"]]
    w0 = cache.widths[s0["id"]]
    i0 = cache.idx(s0["id"])
    hid = {l: cache.hidden[l][i0] for l in cache.layers}

    def one():
        return D.fused_score(gen, enc_masses, hid, w0, cfg.epsilon)

    def five():
        for t in TYPES:
            D.fused_score(specs[t], enc_masses, hid, w0, cfg.epsilon)
        return D.fused_score(gen, enc_masses, hid, w0, cfg.epsilon)

    def bench(fn, iters):
        fn()  # warmup
        t = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t) / iters * 1e3

    n = cfg.n_latency_runs
    ms1 = bench(one, n)
    ms5 = bench(five, n)
    # NOTE: the v3 "< 0.5% overhead vs a single HARM" criterion is a TOTAL
    # latency claim — one shared forward pass dominates, so the percentage
    # only makes sense with the forward included (--with-model). Scoring-only
    # numbers are reported in absolute ms for that reason.
    res = {"single_specialist_ms": ms1, "five_specialists_ms": ms5,
           "scoring_extra_ms": ms5 - ms1,
           "overhead_pct": None, "with_model": False}
    if with_model:
        print("  (forward comparison: loading model ...)")
        from multi_harm_common.chat import encode_sample
        from multi_harm_common.model import forward_signals, load_model
        from config import load_config
        cfg2 = cfg
        model, tok, dev, q = load_model(cfg2)
        encs = [encode_sample(tok, r, cfg.max_seq_len, cfg.tail_len)
                for r in t0]
        cand = cfg.candidate_layers(
            __import__("multi_harm_common.model", fromlist=["get_n_layers"])
            .get_n_layers(model))

        def fwd_one():
            for e in encs:
                sig = forward_signals(model, e, cfg.attn_last_k, cand)
                D.fused_score(gen, sig["masses"], sig["hidden"],
                              sig["widths"], cfg.epsilon)

        def fwd_five():
            for e in encs:
                sig = forward_signals(model, e, cfg.attn_last_k, cand)
                for t in TYPES:
                    D.fused_score(specs[t], sig["masses"], sig["hidden"],
                                  sig["widths"], cfg.epsilon)
                D.fused_score(gen, sig["masses"], sig["hidden"],
                              sig["widths"], cfg.epsilon)

        t1 = bench(fwd_one, 10) / len(encs)
        t5 = bench(fwd_five, 10) / len(encs)
        res["forward_plus_1_ms"] = t1
        res["forward_plus_5_ms"] = t5
        res["forward_overhead_pct"] = (t5 - t1) / t1 * 100
        res["with_model"] = True
        del model
    return res


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-model", action="store_true",
                    help="include the full forward-pass latency comparison")
    ap.add_argument("--calib-sweep", action="store_true",
                    help="re-calibrate every specialist at calib_n in "
                         "{40, 80, 160} (cached signals only; a few minutes)")
    args = ap.parse_args()

    cfg = load_config()
    os.makedirs(EXP(cfg), exist_ok=True)
    cache, df, hstar, specs, gen, meta_cfg = load_all(cfg)
    row1 = load_json(os.path.join(EXP(cfg), "row1_attn_shared.json"))
    row2 = load_json(os.path.join(EXP(cfg), "row2_general.json"))
    hid_base = load_json(os.path.join(EXP(cfg), "baseline_hidden_only.json"))
    if not all([row1, row2, hid_base]):
        print("Run 05 and 06 first (4.8 rows 1-2 + hidden baseline).")
        sys.exit(1)

    print("=== TABLE A — main results (TEST) ===")
    ev_a = table_a(cfg, cache, df, specs, gen, meta_cfg)
    save_json({k: ev_a[k] for k in ev_a if k != "records"},
              os.path.join(EXP(cfg), "tableA_main.json"))
    _ma = ev_a["mean_asr"]
    print(f"  detection {ev_a['detection']:.4f} ASR {ev_a['asr']:.4f} "
          f"FPR {ev_a['fpr']:.4f} mean-ASR "
          f"{'n/a' if _ma is None else format(_ma, '.4f')} "
          f"attr-acc {ev_a['attr_accuracy']}")

    print("\n=== TABLE B — baselines ===")
    tb, _ = table_b(cfg, cache, df, specs, gen, row1, row2, hid_base)
    tb.to_csv(os.path.join(EXP(cfg), "tableB_baselines.csv"), index=False)
    print(tb.groupby("config").agg(auroc_mean=("auroc", "mean")).to_string())

    print("\n=== TABLE C — ablations ===")
    tc = table_c(cfg, cache, df, specs, gen, meta_cfg)
    print(tc.to_string(index=False, max_colwidth=28))

    print("\n=== TABLE D — attribution ===")
    print(f"  accuracy {ev_a['attr_accuracy']}")
    print(pd.DataFrame(ev_a["attr_confusion"]).to_string(index=False))

    print("\n=== TABLE E — unseen attack (v3 §4.5) ===")
    te = table_e_unseen(cfg, cache, df, specs, gen, hstar)
    print(f"  held-out={te['held_out_type']}  ASR_unseen={te['asr_unseen']}  "
          f"mean_seen={te['mean_asr_seen']}  gap={te['gap_points']}  "
          f"within10={te['within_10_points']}")

    print("\n=== §4.3 — signal independence (val, pairwise Pearson) ===")
    pairs, ok43 = sec_43_independence(cfg, cache, df, specs)
    for k, v in pairs.items():
        print(f"  {k:14s} rho={v:+.4f}")
    print(f"  criterion rho<0.5 all pairs: {'MET' if ok43 else 'NOT MET'}")

    # §4.3 asserts "no single head fires for all 4 attack types". The pairwise
    # score correlation above is about the *fused scores*; the head-level claim
    # needs its own table: for every head, its mean mass by attack type over val
    # injected rows, and which type each head prefers.
    try:
        hv = df[(df["split"] == "val") & (df["label"] == 1)]["id"].tolist()
        if not hv:
            raise RuntimeError("no val injected rows in the extracted cache — the "
                               "cross-tab has nothing to summarize")
        sub = cache.subset(hv)
        if not sub["ids"]:
            raise RuntimeError("cache.subset returned no rows for the val ids")
        heads = sorted(sub["masses"][0].keys())        # (layer, head) -> (m_qp, m_qi, m_qq)
        acc = {t: np.zeros(len(heads)) for t in TYPES}
        cnt = {t: 0 for t in TYPES}
        for j, sid in enumerate(sub["ids"]):
            t = str(sub["types"][j])
            if t in acc:
                acc[t] += np.array([sub["masses"][j][h] for h in heads],
                                   dtype=np.float64).sum(axis=1)
                cnt[t] += 1
        tab = pd.DataFrame({t: acc[t] / cnt[t] for t in TYPES if cnt[t]},
                           index=[f"L{l}xH{h}" for (l, h) in heads])
        sel = {t: f"L{specs[t]['head'][0]}xH{specs[t]['head'][1]}"
               for t in TYPES if t in specs}
        # tab.idxmax(axis=0): for each TYPE, the head with the highest mean mass
        argmax_head = tab.idxmax(axis=0).to_dict() if not tab.empty else {}
        won = (tab.idxmax(axis=0).value_counts().to_dict() if not tab.empty else {})
        out43 = {
            "note": ("mean per-head attention mass on val injected rows by attack "
                     "type, over every layer x head. 'head_maximizing' per type is "
                     "which attack each head responds to most; if a single head "
                     "were the argmax for every type the specialization claim "
                     "would be false. n_heads_maximizing_each_type is the "
                     "distribution of argmax winners."),
            "n_val_injected_per_type": cnt,
            "n_heads": int(len(heads)),
            "selected_head_by_type": sel,
            "argmax_head_by_type": argmax_head,
            "n_types_won_per_head": {k: int(v) for k, v in won.items()},
            "max_types_any_head_wins": int(max(won.values())) if won else None,
            "one_head_wins_every_type": bool(won and max(won.values()) == len(TYPES)),
            "selected_head_is_own_type_argmax": {
                t: bool(argmax_head.get(t) == sel[t]) for t in sel},
            "table_csv": "sec43_head_by_type.csv",
        }
        tab.round(4).to_csv(os.path.join(EXP(cfg), "sec43_head_by_type.csv"))
        save_json(out43, os.path.join(EXP(cfg), "sec43_head_by_type.json"))
        if not tab.empty:
            print(f"\n=== §4.3 head x type ({len(heads)} heads) ===")
            print(f"  types won per head: {out43['n_types_won_per_head']} "
                  f"(one head winning all {len(TYPES)} types: "
                  f"{out43['one_head_wins_every_type']} — §4.3's claim is that "
                  f"this is False)")
            print(f"  calibration's head IS its own type's argmax: "
                  f"{out43['selected_head_is_own_type_argmax']}")
    except Exception as e:
        # a skipped diagnostic must never look like a clean run: v3.0's worst bug
        # hid inside `except Exception: return 0.5`
        import traceback
        print(f"  §4.3 head-by-type dump SKIPPED: {type(e).__name__}: {e}")
        print("   " + traceback.format_exc().strip().splitlines()[-3])

    print("\n=== §4.4 — cross-specialist generalization (val injected) ===")
    mat44 = sec_44_cross(cfg, cache, df, specs)
    print(mat44.to_string())

    print("\n=== §4.8 — spine table (TEST AUROCs) ===")
    t48 = sec_48_table(cfg, cache, df, specs, gen, row1, row2)
    for k, v in t48.items():
        print(f"  {k:40s} " + " ".join(f"{t}={v[t]:.4f}" for t in TYPES)
              + f"  spread={v['spread']:.4f}")

    print("\n=== span-width audit (Table A robustness) ===")
    wd = width_diagnostics(cfg, cache, df, hstar)
    for t, v in wd["per_type_width_test"].items():
        print(f"  {t:10s} W_i mean={v['w_i_mean']:6.1f} med={v['w_i_median']:5.1f} "
              f"std={v['w_i_std']:5.1f} (n={v['n']})")
    print(f"  corr(R, W_i): all test {wd['r_width_corr_all_test']:+.4f} | "
          f"injected-only {wd['r_width_corr_injected_test']}")
    _nv = wd.get("naive_sumratio_width_corr_injected_test")
    _iv = wd.get("r_width_corr_injected_test")
    print(f"  width-confound check: deployed head_ratio "
          f"{'n/a' if _iv is None else format(_iv, '+.4f')} vs v3.0 naive "
          f"sum-ratio {'n/a' if _nv is None else format(_nv, '+.4f')} (corr with "
          f"W_i, test injected) — [validation/width_invariance.json]")

    sweep = None
    if args.calib_sweep:
        print("\n=== calibration-size sweep (does specialization survive small n?) ===")
        sweep = calib_size_sweep(cfg, cache, df, hstar)

    print("\n=== latency ===")
    lat = latency_measure(cfg, cache, df, specs, gen, args.with_model)
    save_json(lat, os.path.join(EXP(cfg), "latency.json"))
    print(f"  scoring: 1 spec {lat['single_specialist_ms']:.4f} ms -> "
          f"5 specs {lat['five_specialists_ms']:.4f} ms "
          f"(extra {lat['scoring_extra_ms']:.4f} ms)")
    if lat.get("with_model"):
        print(f"  forward+score: 1 spec {lat['forward_plus_1_ms']:.2f} ms -> "
              f"5 specs {lat['forward_plus_5_ms']:.2f} ms "
              f"(overhead {lat['forward_overhead_pct']:.3f}%)")

    write_summary(cfg, ev_a, tb, tc, te, pairs, ok43, mat44, t48, lat, wd,
                  sweep)
    print(f"\nAll tables written to {EXP(cfg)}/ — see SUMMARY.md")


def write_summary(cfg, ev_a, tb, tc, te, pairs, ok43, mat44, t48, lat, wd,
                  sweep=None):
    qc = load_json(os.path.join(EXP(cfg), "quant_compare.json"), default={})
    L_star_gen = None
    try:
        L_star_gen = load_json(os.path.join(cfg.out_dir, "calib",
                                            "general.json"))["L_star"]
    except Exception:
        pass
    cos_at_L = qc.get("per_layer_cos", {}).get(str(L_star_gen)) \
        or qc.get("per_layer_cos", {}).get(L_star_gen)

    def crit(ok):
        return "MET" if ok else "NOT MET"

    lines = []
    lines.append("# Multi-HARM — Experiment Summary (test split)\n")
    lines.append("## §4.8 spine table (per-attack-type AUROC, test)\n")
    lines.append("| Signal config | " + " | ".join(TYPES) + " | Spread |")
    lines.append("|---|" + "---|" * (len(TYPES) + 1))
    for k, v in t48.items():
        lines.append(f"| {k} | " + " | ".join(f"{v[t]:.4f}" for t in TYPES)
                     + f" | {v['spread']:.4f} |")
    lines.append(f"\n**Primary claim check:** row 3 (fused, specialized) vs row 2 "
                 f"(fused, shared): mean gain "
                 f"{np.mean([t48['fused per-attack specialized (row 3)'][t] - t48['fused shared HARM_general (4.8 r2)'][t] for t in TYPES]):+.4f} "
                 f"-> criterion > 0.03: "
                 f"{crit(np.mean([t48['fused per-attack specialized (row 3)'][t] - t48['fused shared HARM_general (4.8 r2)'][t] for t in TYPES]) > 0.03)}\n")

    lines.append("## Key success criteria (v3)\n")
    lines.append("| Metric | Target | Actual | Status |")
    lines.append("|---|---|---|---|")
    ma = ev_a["mean_asr"]
    lines.append(f"| Multi-HARM mean ASR | < 8% | "
                 f"{'n/a' if ma is None else format(ma, '.4f')} | "
                 f"{crit(ma is not None and ma < 0.08)} |")
    mean_gain = np.mean([t48['fused per-attack specialized (row 3)'][t]
                         - t48['fused shared HARM_general (4.8 r2)'][t]
                         for t in TYPES])
    lines.append(f"| Specialized vs shared fusion gain (§4.8 r3-r2) | > 3 pts | "
                 f"{mean_gain:.4f} | {crit(mean_gain > 0.03)} |")
    lines.append(f"| Mean FPR | < 5% | {ev_a['fpr']:.4f} | {crit(ev_a['fpr'] < 0.05)} |")
    lines.append(f"| — calibration-set sizes | >= 1/fpr_budget clean samples | "
                 f"{load_json(os.path.join(cfg.out_dir, 'calib', 'specialists.json'))['naive']['calib']['n_clean']}"
                 f" clean per specialist | "
                 f"{crit(load_json(os.path.join(cfg.out_dir, 'calib', 'specialists.json'))['naive']['calib']['n_clean'] >= int(4 / cfg.target_fpr))}"
                 f" (see the n_neg NOTE printed by 07) |")
    acc = ev_a['attr_accuracy']
    lines.append(f"| Attribution accuracy | > 75% | "
                 f"{acc if acc is None else format(acc, '.4f')} | "
                 f"{crit(acc is not None and acc > 0.75)} |")
    lines.append(f"| Pairwise specialist-score rho | < 0.5 | max {max(pairs.values()):.4f} | {crit(ok43)} |")
    lat_pct = lat.get("forward_overhead_pct", lat.get("overhead_pct"))
    lines.append(f"| Latency overhead vs single specialist | < 0.5% | "
                 f"{'n/a' if lat_pct is None else format(lat_pct, '.3f') + ' %'} | "
                 f"{'n/a' if lat_pct is None else crit(lat_pct < 0.5)} "
                 f"{'(run 09 --with-model for the forward-included number)' if lat_pct is None else ''} |")
    def _f4(x):
        return "n/a" if not isinstance(x, (int, float)) else format(float(x), ".4f")
    lines.append(f"| Unseen-attack ASR | within 10 pts of seen | "
                 f"{_f4(te['asr_unseen'])} vs {_f4(te['mean_asr_seen'])} "
                 f"(gap {_f4(te['gap_points'])}) | "
                 f"{'n/a' if te['within_10_points'] is None else crit(te['within_10_points'])} |")
    if qc.get("applicable"):
        lines.append(f"| fp-ref vs 4-bit correlation at L* | > 0.9 | "
                     f"{cos_at_L} (L*={L_star_gen}) | "
                     f"{crit(cos_at_L is not None and cos_at_L > 0.9)} |")
    else:
        lines.append("| fp-ref vs 4-bit correlation at L* | > 0.9 | n/a "
                     "(not applicable on this device) | n/a |")

    lines.append("\n## Span-width audit (Table A robustness)\n")
    lines.append("Injection span token counts (test) — shows `combined` payloads "
                 "are much longer than `naive`; R is span-width-invariant "
                 "(per-column-mean ratio), and the residual R↔width "
                 "correlation is reported here.\n")
    lines.append("| type | n | W_i mean | W_i median | W_i std |")
    lines.append("|---|---|---|---|---|")
    for t, v in wd["per_type_width_test"].items():
        lines.append(f"| {t} | {v['n']} | {v['w_i_mean']:.1f} | "
                     f"{v['w_i_median']:.1f} | {v['w_i_std']:.1f} |")
    inj_corr = wd["r_width_corr_injected_test"]
    nv = wd.get("naive_sumratio_width_corr_injected_test")
    lines.append(f"\ncorr(R, W_i) on test: all samples "
                 f"{wd['r_width_corr_all_test']:+.4f} | injected-only "
                 f"{inj_corr if inj_corr is None else format(inj_corr, '+.4f')} "
                 f"\n\nWidth invariance, measured against the estimator it replaced "
                 f"(corr with W_i, injected test): deployed head_ratio "
                 f"{inj_corr if inj_corr is None else format(inj_corr, '+.4f')}, "
                 f"v3.0 naive sum-ratio {nv if nv is None else format(nv, '+.4f')} "
                 f"— {'confound demonstrated: the old statistic was width-driven' if (nv is not None and inj_corr is not None and abs(nv) > abs(inj_corr) + 0.10) else 'not demonstrated on this dataset (width range too narrow, or both estimators flat)'}"
                 f" (see out/validation/width_invariance.json)")

    x44 = load_json(os.path.join(EXP(cfg), "sec44_cross_val.json"), default={})
    if x44:
        lines.append("\n## §4.4 — cross-specialist firing (honest attribution check)\n")
        lines.append(f"Injected val samples whose highest fused score came from the "
                     f"matching type specialist: "
                     f"{'n/a' if x44.get('own_type_fraction') is None else format(x44['own_type_fraction'], '.4f')} "
                     f"(n={x44.get('n')}). Quote this next to Table D: Table D's "
                     f"argmax accuracy is upward-biased by construction, since each "
                     f"specialist is calibrated on its own attack type.\n")
        lines.append("| true \\ predicted | " + " | ".join(TYPES) + " |")
        lines.append("|---|" + "---|" * len(TYPES))
        for t in TYPES:
            row = x44.get("matrix", {}).get(t, {})
            lines.append(f"| {t} | " + " | ".join(str(row.get(u, 0)) for u in TYPES) + " |")

    if sweep is not None:
        lines.append("\n## Calibration-size sweep (--calib-sweep)\n")
        lines.append("| calib_n | mean fused AUROC (calib) | mean fused (held-out "
                     "calib) | mean fused (test, per type vs clean) |")
        lines.append("|---|---|---|---|")
        g = (sweep.groupby("calib_n")[["fused_calib", "fused_heldout",
                                       "fused_test"]].mean())
        for n_, r in g.iterrows():
            lines.append(f"| {n_} | {r['fused_calib']:.4f} | "
                         f"{r['fused_heldout']:.4f} | {r['fused_test']:.4f} |")
        lines.append("\nIf the test column collapses as calib_n shrinks, the "
                     "specialization gain is calibration-limited, not attack-type "
                     "structure — state that instead of the headline.\n")

    lines.append("\n## Tables\n")
    lines.append("- Table A: out/experiments/tableA_main.csv")
    lines.append("- Table B: out/experiments/tableB_baselines.csv")
    lines.append("- Table C: out/experiments/tableC_ablations.csv")
    lines.append("- Table D: attribution confusion in test_records.csv / tableA_main.json")
    lines.append("- Table E: out/experiments/tableE_unseen.json")
    lines.append("- §4.3: out/experiments/sec43_pearson_val.csv")
    lines.append("- §4.4: out/experiments/sec44_cross_val.csv")
    lines.append("- §4.8: out/experiments/table_48.json (also above)")
    lines.append("- latency: out/experiments/latency.json")
    lines.append("- quant compare: out/experiments/quant_compare.json")
    lines.append("- span-width audit: out/experiments/span_width.json")
    lines.append("- §4.4 (honest attribution): out/experiments/sec44_cross_val.json")
    lines.append("- calibration-size sweep: out/experiments/calib_size_sweep.csv "
                 "(only with --calib-sweep)")
    with open(os.path.join(EXP(cfg), "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
