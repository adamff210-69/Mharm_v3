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


def _balanced_split(labels: np.ndarray, rng, frac: float = 0.5):
    """Stratified fit/eval split that keeps BOTH classes in the eval half.

    Required for honest AUROC reporting: if we select the best head / layer on
    the same rows we then evaluate it on, the reported AUROC is in-sample
    selection AUC (typically inflated to ~1.0 even with no real signal). The
    head-selection and half-split tables must use data not used for selection.
    """
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    fit, ev = [], []
    for cls in sorted(np.unique(labels)):
        ci = idx[labels == cls].copy()
        k = max(1, int(np.floor(len(ci) * frac)))
        if len(ci) - k == 0 and len(ci) > 1:
            k = len(ci) - 1
        rng.shuffle(ci)
        fit.append(ci[:k])
        ev.append(ci[k:])
    fit = np.concatenate(fit) if fit else np.array([], dtype=int)
    ev = np.concatenate(ev) if ev else np.array([], dtype=int)
    return fit, ev


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

    # Select H* on a stratified fit half, evaluate on the held-out half so the
    # reported pooled AUROC is NOT in-sample selection AUC (which is why the
    # old 04 table could show 0.50 for every head on a trivial pilot).
    rng = np.random.default_rng(cfg.seed + 11)
    fit_idx, eval_idx = _balanced_split(sub["labels"], rng, frac=0.5)
    fit_ids = [ids[i] for i in fit_idx]
    eval_ids = [ids[i] for i in eval_idx]
    sub_fit = cache.subset(fit_ids)
    sub_eval = cache.subset(eval_ids)

    head_res = S.select_h_star(sub_fit["masses"], sub_fit["widths"],
                               sub_fit["labels"], attn_layers, n_heads,
                               top_k=cfg.top_k_heads, eps=cfg.epsilon)
    head = tuple(head_res["best_head"])
    r_eval = np.array([S.head_ratio(m[head], cfg.epsilon, w)
                       for m, w in zip(sub_eval["masses"], sub_eval["widths"])])
    eval_auroc = float(auroc(sub_eval["labels"], r_eval))

    res = dict(head_res)
    res["best_auroc"] = eval_auroc
    res["select_auroc"] = head_res["best_auroc"]  # selection-set AUROC (report separately, not as the truth)
    res["calib_ids"] = ids
    res["eval_ids"] = eval_ids
    res["n_clean"] = int((sub["labels"] == 0).sum())
    res["n_inj"] = int((sub["labels"] == 1).sum())
    res["n_eval_clean"] = int((sub_eval["labels"] == 0).sum())
    res["n_eval_inj"] = int((sub_eval["labels"] == 1).sum())
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

    # Stratified fit/eval split for the residual half. The probe is trained only
    # on the fit subset and all reported half-split AUROCs use the eval subsets,
    # so the "hidden = 1.0" numbers cannot be an in-sample artifact.
    rng_probe = np.random.default_rng(seed if seed is not None else cfg.seed + 5)
    fit_idx, eval_idx = _balanced_split(labels, rng_probe, cfg.probe_fit_frac)
    hid_fit = [{l: sub["hid"][l][i] for l in cache.layers} for i in fit_idx]
    labels_fit = labels[fit_idx]
    labels_eval = labels[eval_idx]

    # --- 1) L* (per-specialist layer selection) -----------------------------
    lres = S.select_l_star(hid_fit, labels_fit, cache.layers,
                           fit_frac=cfg.probe_fit_frac)
    L = lres["best_layer"]

    # --- 2) probe at L* -------------------------------------------------------
    probe, probe_auroc_fit, probe_auroc_eval = S.fit_probe(
        hid_fit, labels_fit, L, cfg.probe_fit_frac)
    p_scores = np.array([S.probe_probs(hid_sample[i][L], probe) for i in range(n)])

    # --- 3) h_base (mean clean embedding at L*) + cosine diagnostic ----------
    clean_idx = [i for i in range(n) if labels[i] == 0]
    h_base = (np.mean([hid_sample[i][L] for i in clean_idx], axis=0)
              if clean_idx else np.zeros_like(hid_sample[0][L]))
    cos = np.array([float(np.dot(hid_sample[i][L], h_base) /
                          (np.linalg.norm(hid_sample[i][L]) *
                           np.linalg.norm(h_base) + 1e-12)) for i in range(n)])

    # --- 4) attention half: per-specialist H*_s and shared H* ----------------
    # Select the specialist head on the SAME stratified fit half used by the
    # probe/L* above; evaluate the half-split AUROC on the held-out half. This
    # is what makes the "attention half" number interpretable (otherwise
    # selecting the best of 128 heads on the same ~30 calibration rows
    # guarantees near-1.0 AUROC).
    widths = sub["widths"]
    sub_fit = cache.subset([ids[int(i)] for i in fit_idx])
    sub_eval = cache.subset([ids[int(i)] for i in eval_idx])
    head_res = S.select_h_star(sub_fit["masses"], sub_fit["widths"],
                               sub_fit["labels"], attn_layers, n_heads,
                               top_k=cfg.top_k_heads, eps=cfg.epsilon)
    head = head_res["best_head"]
    if use_shared_head:
        head = hstar["best_head"]
    r_head = np.array([S.head_ratio(masses[i][tuple(head)], cfg.epsilon,
                                    widths[i]) for i in range(n)])
    r_shared = np.array([S.head_ratio(masses[i][tuple(hstar["best_head"])],
                                      cfg.epsilon, widths[i])
                         for i in range(n)])
    r_head_eval = np.array([S.head_ratio(sub_eval["masses"][i][tuple(head)],
                                         cfg.epsilon, sub_eval["widths"][i])
                            for i in range(len(sub_eval["ids"]))])
    r_shared_eval = np.array([S.head_ratio(sub_eval["masses"][i][tuple(hstar["best_head"])],
                                           cfg.epsilon, sub_eval["widths"][i])
                              for i in range(len(sub_eval["ids"]))])
    att_head_eval = float(auroc(sub_eval["labels"], r_head_eval))
    att_shared_eval = float(auroc(sub_eval["labels"], r_shared_eval))

    # --- 5) fusion weight alpha ------------------------------------------------
    # Select alpha on the held-out half (not the rows a probe / head was fitted
    # on); use the chosen standardization stats for deployment as well so the
    # calibration and inference scores use the same transform.
    fres = S.choose_alpha(r_head[eval_idx], p_scores[eval_idx],
                          labels_eval, step=cfg.alpha_step)
    a = fres["alpha"]
    r_mu, r_sd, p_mu, p_sd = (fres["r_mu"], fres["r_sd"],
                              fres["p_mu"], fres["p_sd"])
    zr = (r_head - r_mu) / max(r_sd, 1e-12)
    zp = (p_scores - p_mu) / max(p_sd, 1e-12)
    fused = a * zr + (1 - a) * zp
    fused_eval_auroc = float(auroc(labels_eval, fused[eval_idx]))

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
        # v3 Phase 3 addition — the two halves logged BEFORE fusion. Attention,
        # hidden and fused are all evaluated on a HELD-OUT split (not the rows
        # used to select the head / train the probe / tune alpha), so the
        # half-split table is not an in-sample artifact.
        "auroc": {
            "att_head": att_head_eval,
            "att_shared": att_shared_eval,
            "att_head_selection": float(auroc(labels, r_head)),
            "hid": probe_auroc_eval,
            "fused": fused_eval_auroc,
            "cos_diag": float(auroc(labels_eval, -cos[eval_idx])),
        },
        "calib": {
            "n_samples": n,
            "n_clean": int((labels == 0).sum()),
            "n_inj": int((labels == 1).sum()),
            "n_attn_fit": len(fit_idx),
            "n_attn_eval": len(eval_idx),
            "ids": ids,
            "l_star_curve": lres["per_layer"],
            "head_curve": head_res["all"],
            "hstar_best_auroc": float(head_res["best_auroc"]),
            "probe_fit_auroc": float(probe_auroc_fit),
            "probe_eval_auroc": float(probe_auroc_eval),
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
