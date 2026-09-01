"""Specialist scoring and meta-decision logic.

A *specialist* is a fully-calibrated dict (JSON-serializable, saved under
out/calib/). Scoring is pure numpy dot products on cached tensors — this is
the "cheap dot product" that makes 5 specialists cost ~0% extra over one
forward pass (v3 shared-computation claim).
"""
from __future__ import annotations

import numpy as np

from .signals import fuse, head_ratio, probe_probs


# ---------------------------------------------------------------------------
# Specialist spec (dict) schema — produced by calibrate.calibrate_specialist
# ---------------------------------------------------------------------------
# {
#  "name": "topic" | "naive" | "fake" | "combined" | "general",
#  "attack_type": one of the four | None,
#  "head": [layer, head_idx],          # per-specialist best head (H*_s)
#  "head_shared": [layer, head_idx],   # pooled H* (for comparability / general)
#  "top_k": [[l, h], auroc, ...],
#  "L_star": int,
#  "probe": {"layer": int, "coef": [...], "bias": float, "mean": [...], "std": [...]},
#  "h_base": [...],                    # mean clean embedding at L*_s
#  "alpha": float, "r_mu": float, "r_sd": float, "p_mu": float, "p_sd": float,
#  "theta": float, "fpr_budget": float,
#  "auroc": {"att_head": f, "att_shared": f, "hid": f, "fused": f, "cos_diag": f},
#  "calib": {...}                      # alpha curve, theta sweep info, sizes
# }

def score_attention(spec: dict, masses: dict, widths: tuple,
                    heads=None, eps: float = 1e-6) -> float:
    """z-standardized attention-half score using the spec's head(s).

    widths: (W_p, W_i, W_q) for THIS sample — required for the
    span-width-invariant ratio (see signals.head_ratio).
    """
    heads = heads or [tuple(spec["head"])]
    rs = np.array([head_ratio(masses[h], eps, widths) for h in heads])
    r = float(rs.mean())
    return (r - spec["r_mu"]) / max(spec["r_sd"], 1e-12)


def score_hidden(spec: dict, hidden: dict) -> float:
    """z-standardized residual-half score P(injection) at L*_s."""
    p = probe_probs(np.asarray(hidden[spec["L_star"]], dtype=float), spec["probe"])
    return (float(p) - spec["p_mu"]) / max(spec["p_sd"], 1e-12)


def fused_score(spec: dict, masses: dict, hidden: dict, widths: tuple,
                eps: float = 1e-6) -> dict:
    zr = score_attention(spec, masses, widths, eps=eps)
    zp = score_hidden(spec, hidden)
    a = spec["alpha"]
    return {"s": a * zr + (1 - a) * zp, "zr": zr, "zp": zp,
            "p_inj": float(probe_probs(np.asarray(hidden[spec["L_star"]], dtype=float),
                                       spec["probe"]))}


# ---------------------------------------------------------------------------
# Meta-decision
# ---------------------------------------------------------------------------

def meta_decision(spec_scores: dict, specs: dict, general_score: float | None,
                  general_theta: float | None, mode: str,
                  global_theta: float | None = None,
                  eps: float = 1e-6):
    """Combine specialist fused scores into INJECTED/SAFE + attribution.

    spec_scores: {specialist_name: fused_score_dict} for the 4 type specialists.
    Returns dict: decision, attribution, fired, s_max, margin, scores.
    """
    scores = {n: d["s"] for n, d in spec_scores.items()}
    if mode == "global_max":
        all_scores = dict(scores)
        if general_score is not None:
            all_scores["general"] = general_score
        s_max = max(all_scores.values())
        decision = s_max > (global_theta if global_theta is not None else 1e9)
        attribution = max(all_scores, key=all_scores.get) if decision else None
        top2 = sorted(all_scores.values(), reverse=True)[:2]
        margin = top2[0] - top2[1] if len(top2) > 1 else top2[0]
        return {"decision": bool(decision), "attribution": attribution,
                "fired": [n for n, v in all_scores.items() if v > 0] if decision else [],
                "s_max": float(s_max), "margin": float(margin), "scores": scores}

    # default: per_spec — OR of individually-thresholded specialists; the
    # general specialist is the fallback when no type specialist fires.
    fired = [n for n, s in scores.items() if s > specs[n]["theta"]]
    if fired:
        attribution = max(fired, key=lambda n: scores[n])
        decision = True
    elif general_score is not None and general_score > (general_theta or 1e9):
        attribution = "general"
        decision = True
    else:
        attribution = None
        decision = False
    s_max = max(scores.values()) if scores else 0.0
    vals = sorted(scores.values(), reverse=True)
    margin = vals[0] - vals[1] if len(vals) > 1 else vals[0]
    return {"decision": decision, "attribution": attribution,
            "fired": fired, "s_max": float(s_max), "margin": float(margin),
            "scores": scores}


# ---------------------------------------------------------------------------
# Batch scoring on cached signals (no model needed)
# ---------------------------------------------------------------------------

def score_spec_on_split(spec: dict, df, cache, split: str, eps: float = 1e-6
                        ) -> dict:
    """Fused scores for one specialist on all samples of a split."""
    rows = df[df["split"] == split].reset_index(drop=True)
    s = cache.subset(rows["id"].tolist())
    n = len(s["ids"])
    zr = np.array([(head_ratio(s["masses"][i][tuple(spec["head"])], eps,
                               s["widths"][i]) - spec["r_mu"])
                   / max(spec["r_sd"], 1e-12) for i in range(n)])
    zp = np.array([((probe_probs(s["hid"][spec["L_star"]][i], spec["probe"])
                     - spec["p_mu"]) / max(spec["p_sd"], 1e-12)) for i in range(n)])
    fused = fuse(spec["alpha"], zr, zp)
    return {"ids": s["ids"], "labels": s["labels"], "types": s["types"],
            "s": fused, "zr": zr, "zp": zp}


def evaluate_meta(type_specs: list[dict], general_spec: dict | None, df, cache,
                  split: str, mode: str, cfg, global_theta: float | None = None
                  ) -> dict:
    """Run the full Multi-HARM pipeline (5 specialists + meta) on one split.

    Returns per-sample records and aggregate metrics:
      overall / per-type: detection rate, ASR (1 - TPR), FPR, F1
      attribution accuracy on detected injected samples + confusion pairs.
    """
    rows = df[df["split"] == split].reset_index(drop=True)
    per_spec = {sp["name"]: score_spec_on_split(sp, df, cache, split, cfg.epsilon)
                for sp in type_specs}
    gen = (score_spec_on_split(general_spec, df, cache, split, cfg.epsilon)
           if general_spec is not None else None)
    labels = per_spec[type_specs[0]["name"]]["labels"]
    types = per_spec[type_specs[0]["name"]]["types"]

    records = []
    for i, sid in enumerate(rows["id"].tolist()):
        spec_scores = {n: {"s": float(ps["s"][i])} for n, ps in per_spec.items()}
        gen_score = float(gen["s"][i]) if gen is not None else None
        gen_theta = general_spec["theta"] if general_spec is not None else None
        d = meta_decision(spec_scores,
                          {n: {"theta": type_specs[k]["theta"]}
                           for k, n in enumerate([sp["name"] for sp in type_specs])},
                          gen_score, gen_theta, mode, global_theta, cfg.epsilon)
        records.append({
            "id": sid, "attack_type": types[i], "label": int(labels[i]),
            **{f"s_{n}": v["s"] for n, v in spec_scores.items()},
            "s_general": gen_score,
            "decision": d["decision"], "attribution": d["attribution"],
            "fired": "|".join(d["fired"]), "margin": d["margin"],
            "s_max": d["s_max"],
        })

    import pandas as pd
    R = pd.DataFrame(records)
    y = R["label"].to_numpy()
    pred = R["decision"].astype(int).to_numpy()
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    det = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * det / max(1e-12, prec + det)

    per_type = {}
    for at in list(dict.fromkeys(types.tolist())):
        m = (y == 1) & (types == at)
        if not m.any():
            continue
        t = int(np.sum(pred[m] == 1))
        per_type[str(at)] = {
            "n": int(m.sum()),
            "detection": t / max(1, int(m.sum())),
            "asr": 1.0 - t / max(1, int(m.sum())),
        }
    asr_vals = [v["asr"] for v in per_type.values()]

    # attribution accuracy: on detected injected samples, does argmax attribution
    # match the true attack type? (general/None counted as misses)
    det_inj = R[(R["label"] == 1) & (R["decision"])]
    attr_acc = None
    confusion = []
    if len(det_inj):
        hits = int((det_inj["attribution"] == det_inj["attack_type"]).sum())
        attr_acc = hits / len(det_inj)
        confusion = det_inj.groupby(["attack_type", "attribution"]).size() \
                           .reset_index(name="n").to_dict("records")

    return {"records": R, "n": len(R),
            "detection": det, "asr": 1.0 - det, "fpr": fpr,
            "precision": prec, "f1": f1,
            "per_type": per_type,
            "mean_asr": float(np.mean(asr_vals)) if asr_vals else None,
            "attr_accuracy": attr_acc, "attr_confusion": confusion}
