"""Specialist calibration orchestration (v3 Phase 3 order:
L* -> probe -> h_base -> alpha -> theta), including the half-split AUROC
logging required by the v3 Phase 3 addition and the pooled (shared) H*
calibration of v3 §2.1.

Everything here runs on *cached* signal tensors (no model forward passes), so
re-calibration after a design tweak takes seconds, not hours.
"""
from __future__ import annotations

import numpy as np

from . import signals as S
from .metrics import auroc, pearson
from .sigcache import SigCache

TYPES = ["topic", "naive", "fake", "combined"]


# ---------------------------------------------------------------------------
# Deterministic stratified calibration sampling (train split only)
# ---------------------------------------------------------------------------

def select_calib_ids(df, seed: int, n_clean: int, n_inj: int,
                     attack_type: str | None) -> list[str]:
    """Balanced clean+injected calibration ids. attack_type=None -> injected
    drawn proportionally across all four types (pooled)."""
    rng = np.random.default_rng(seed)
    tr = df[df["split"] == "train"]
    cl = tr[tr["label"] == 0]["id"].tolist()
    inj = tr[tr["label"] == 1]
    if attack_type is None:
        per = max(1, n_inj // len(TYPES))
        sel_inj = []
        for at in TYPES:
            ids = inj[inj["attack_type"] == at]["id"].tolist()
            if ids:
                sel_inj.extend(rng.choice(ids, size=min(per, len(ids)),
                                          replace=False).tolist())
    else:
        ids = inj[inj["attack_type"] == attack_type]["id"].tolist()
        sel_inj = (rng.choice(ids, size=min(n_inj, len(ids)), replace=False)
                   .tolist() if ids else [])
    sel_cl = (rng.choice(cl, size=min(n_clean, len(cl)), replace=False).tolist()
              if cl else [])
    return sel_cl + sel_inj


# ---------------------------------------------------------------------------
# Pooled H* (v3 §2.1 / 4.8 row 1)
# ---------------------------------------------------------------------------

def calibrate_pooled_hstar(df, cache: SigCache, cfg, n_heads: int) -> dict:
    tr = df[df["split"] == "train"]
    half = max(8, cfg.calib_h_samples // 2)
    ids = select_calib_ids(tr, cfg.seed, half, half, None)
    sub = cache.subset(ids)
    attn_layers = sorted({l for m in sub["masses"] for (l, _) in m})
    res = S.select_h_star(sub["masses"], sub["widths"], sub["labels"],
                          attn_layers, n_heads,
                          top_k=cfg.top_k_heads, eps=cfg.epsilon)
    res["calib_ids"] = ids
    res["n_clean"] = int((sub["labels"] == 0).sum())
    res["n_inj"] = int((sub["labels"] == 1).sum())
    res["n_heads"] = n_heads
    res["attn_layers"] = attn_layers
    return res


# ---------------------------------------------------------------------------
# One specialist (v3 Phase 3)
# ---------------------------------------------------------------------------

def calibrate_specialist(name: str, df, cache: SigCache, cfg, hstar: dict,
                         use_shared_head: bool = False,
                         seed: int | None = None) -> dict:
    """Full calibration for one specialist.

    use_shared_head=True -> HARM_general (4.8 row 2): attention half uses the
    *pooled* H* so the row is a true "shared calibration" configuration.
    """
    attack_type = None if name == "general" else name
    tr = df[df["split"] == "train"]
    half = max(8, cfg.calib_per_specialist // 2)
    ids = select_calib_ids(tr, seed if seed is not None else
                           (cfg.seed + (0 if name == "general" else 7)),
                           half, half, attack_type)
    sub = cache.subset(ids)
    labels, masses = sub["labels"], sub["masses"]
    attn_layers = sorted({l for m in masses for (l, _) in m})
    n_heads = max(h for (_, h) in masses[0]) + 1
    n = len(ids)
    hid_sample = [{l: sub["hid"][l][i] for l in cache.layers} for i in range(n)]

    # --- 1) L* (per-specialist layer selection) -----------------------------
    lres = S.select_l_star(hid_sample, labels, cache.layers,
                           fit_frac=cfg.probe_fit_frac)
    L = lres["best_layer"]

    # --- 2) probe at L* -------------------------------------------------------
    probe, probe_auroc_fit = S.fit_probe(hid_sample, labels, L,
                                         cfg.probe_fit_frac)
    p_scores = np.array([S.probe_probs(hid_sample[i][L], probe) for i in range(n)])

    # --- 3) h_base (mean clean embedding at L*) + cosine diagnostic ----------
    clean_idx = [i for i in range(n) if labels[i] == 0]
    h_base = (np.mean([hid_sample[i][L] for i in clean_idx], axis=0)
              if clean_idx else np.zeros_like(hid_sample[0][L]))
    cos = np.array([float(np.dot(hid_sample[i][L], h_base) /
                          (np.linalg.norm(hid_sample[i][L]) *
                           np.linalg.norm(h_base) + 1e-12)) for i in range(n)])

    # --- 4) attention half: per-specialist H*_s and shared H* ----------------
    widths = sub["widths"]
    head_res = S.select_h_star(masses, widths, labels, attn_layers, n_heads,
                               top_k=cfg.top_k_heads, eps=cfg.epsilon)
    head = head_res["best_head"]
    if use_shared_head:
        head = hstar["best_head"]
    r_head = np.array([S.head_ratio(masses[i][tuple(head)], cfg.epsilon,
                                    widths[i]) for i in range(n)])
    r_shared = np.array([S.head_ratio(masses[i][tuple(hstar["best_head"])],
                                      cfg.epsilon, widths[i])
                         for i in range(n)])

    # --- 5) fusion weight alpha ------------------------------------------------
    fres = S.choose_alpha(r_head, p_scores, labels, step=cfg.alpha_step)
    a = fres["alpha"]
    zr, r_mu, r_sd = S.zscore(r_head)
    zp, p_mu, p_sd = S.zscore(p_scores)
    fused = a * zr + (1 - a) * zp

    # --- 6) threshold theta (FPR-budget constrained) ---------------------------
    budget = cfg.target_fpr if name == "general" else cfg.target_fpr / 4.0
    tres = S.choose_theta(fused, labels, budget)

    spec = {
        "name": name,
        "attack_type": attack_type,
        "head": head,
        "head_shared": hstar["best_head"],
        "top_k": head_res["top_k"],
        "L_star": L,
        "probe": probe,
        "h_base": h_base.tolist(),
        "alpha": a,
        "r_mu": r_mu, "r_sd": r_sd, "p_mu": p_mu, "p_sd": p_sd,
        "theta": tres["theta"],
        "fpr_budget": budget,
        # v3 Phase 3 addition — the two halves logged BEFORE fusion:
        "auroc": {
            "att_head": float(auroc(labels, r_head)),
            "att_shared": float(auroc(labels, r_shared)),
            "hid": float(auroc(labels, p_scores)),
            "fused": float(auroc(labels, fused)),
            "cos_diag": float(auroc(labels, -cos)),
        },
        "calib": {
            "n_samples": n,
            "n_clean": int((labels == 0).sum()),
            "n_inj": int((labels == 1).sum()),
            "ids": ids,
            "l_star_curve": lres["per_layer"],
            "head_curve": head_res["all"],
            "hstar_best_auroc": float(head_res["best_auroc"]),
            "probe_fit_auroc": float(probe_auroc_fit),
            "alpha_curve": fres["curve"],
            "theta_info": tres,
            "pearson_half": float(pearson(r_head, p_scores)),
        },
    }
    return spec


# ---------------------------------------------------------------------------
# Unseen-attack re-calibration (v3 §4.5) — cheap, runs on cached signals
# ---------------------------------------------------------------------------

def recalibrate_general_without(df, cache: SigCache, cfg, hstar: dict,
                                excluded_type: str) -> dict:
    """HARM_general recalibrated with all training samples of one attack type
    removed (for the held-out/unseen-attack test)."""
    df2 = df[~(df["attack_type"] == excluded_type)].copy()
    return calibrate_specialist("general", df2, cache, cfg, hstar,
                                use_shared_head=True, seed=cfg.seed + 31)
