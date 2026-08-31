"""Signal definitions: attention-ratio signal R, residual-stream probe
P(injection), H* / L* selection, fusion weight alpha, threshold theta.

Design decisions (documented because the v2 scaffold was not available; the
v3 spec fixes the *structure* — per-specialist L*, probe, h_base, alpha,
theta — but not every formula):

* R(l, h) = (m_qi/W_i) / (m_qp/W_p + eps): the per-token attention intensity
  on the injection region relative to the per-token intensity on the whole
  passage. m_qi and m_qp are summed over (query rows x span columns);
  dividing by the span widths makes the ratio invariant to injection length
  (so per-type comparisons are not confounded by payload length — `combined`
  payloads are much longer than `naive` ones) and to query length. For clean
  samples the "injection span" is the last ``tail_len`` tokens of the passage
  (a pseudo-injection region), so the signal is defined for both classes and
  measures whether query-attention intensity is disproportionately high on
  the tail of the passage — exactly where injected instructions sit in this
  dataset. This is the operationalization of the "distraction effect"
  reported by Attention Tracker (Hung et al.), adapted to our passage/query
  layout.
* P(injection) = sigmoid(w . (h - mu) / sigma + b): a linear probe on the
  last-token hidden state at L*, standardized features, trained with logistic
  regression. h_base (mean clean embedding at L*) is stored and used for the
  cosine-similarity diagnostic.
* Specialist fused score: S = alpha * z(R) + (1 - alpha) * z(P), with
  per-specialist alpha (grid search on calibration AUROC) and per-specialist
  threshold theta (FPR-budget constrained).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .metrics import auroc


# ---------------------------------------------------------------------------
# Attention-half signal
# ---------------------------------------------------------------------------

def head_ratio(mass: tuple, eps: float = 1e-6, widths: tuple | None = None) -> float:
    """Attention-ratio signal for one (layer, head).

    SPAN-WIDTH INVARIANCE (v3 review fix). Masses are SUMS over
    (query rows x span columns), so ANY naive ratio is confounded by span
    length: `combined` payloads run several times longer than `naive` ones,
    and a raw m_qi/m_qp ratio grows mechanically with W_i even at constant
    per-token attention. (Using the whole passage in the denominator is NOT
    sufficient either — the injection region dilutes its own denominator.)

    R = (m_qi/W_i) / ((m_qp - m_qi)/(W_p - W_i) + eps)

    i.e. the per-token attention INTENSITY on the injection region relative
    to the per-token intensity on the NON-injection part of the passage
    (the "body"). At constant per-token intensities R is EXACTLY independent
    of W_i, W_p and the query length (query-row count cancels: identical
    rows in numerator and denominator). R > 1 means the model attends more
    intensely per token to the injection region than to the passage body —
    the operationalized "distraction effect" (Attention Tracker) for this
    passage/query layout. For clean samples the injection region is the
    last `tail_len` tokens of the passage, so the same quantity is defined
    for both classes.

    `mass` = (m_qp, m_qi, m_qq) RAW column sums; `widths` = (W_p, W_i, W_q)
    in tokens — REQUIRED; raise if omitted, so the invariant form can never
    be silently replaced by the naive ratio.
    """
    if widths is None:
        raise ValueError("head_ratio requires widths=(W_p, W_i, W_q): the "
                         "naive sum-ratio is span-length confounded")
    m_qp, m_qi, _m_qq = mass
    w_p, w_i = max(1, int(widths[0])), max(1, int(widths[1]))
    w_r = max(1, w_p - w_i)
    m_r = max(0.0, m_qp - m_qi)
    return (m_qi / w_i) / (m_r / w_r + eps)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    if sd < 1e-12:
        sd = 1.0
    return (x - mu) / sd, mu, sd


# ---------------------------------------------------------------------------
# H* selection (v3 §2.1 — per-head AUROC, 150-200 calibration samples)
# ---------------------------------------------------------------------------

def select_h_star(masses_by_sample: list[dict], widths_by_sample: list[tuple],
                  labels: np.ndarray, attn_layers: list[int], n_heads: int,
                  top_k: int = 5, eps: float = 1e-6) -> dict:
    """Per (layer, head) AUROC of R (span-invariant) on the calibration set.

    masses_by_sample: one dict per sample, {(l, h): [m_qp, m_qi, m_qq]} raw
    column sums; widths_by_sample: (W_p, W_i, W_q) per sample.
    """
    n = len(masses_by_sample)
    cand = [(l, h) for l in attn_layers for h in range(n_heads)]
    scores = np.zeros((n, len(cand)))
    for i, m in enumerate(masses_by_sample):
        for j, (l, h) in enumerate(cand):
            scores[i, j] = head_ratio(m[(l, h)], eps, widths_by_sample[i])
    y = (labels == 1).astype(float)
    aurocs = np.array([auroc(y, scores[:, j]) for j in range(len(cand))])
    order = np.argsort(-aurocs)
    best = int(order[0])
    return {
        "best_head": [int(cand[best][0]), int(cand[best][1])],
        "best_auroc": float(aurocs[best]),
        "top_k": [
            [[int(cand[i][0]), int(cand[i][1])], float(aurocs[i])]
            for i in order[:top_k]
        ],
        "all": {f"{l}x{h}": float(a) for (l, h), a in zip(cand, aurocs)},
    }


# ---------------------------------------------------------------------------
# Residual-half signal: L* selection + linear probe
# ---------------------------------------------------------------------------

def _stratified_split(labels: np.ndarray, rng, fit_frac: float = 0.7):
    """Stratified fit/eval split. Keeps both classes in both halves whenever
    possible; a plain shuffle can put the whole held-out slice in one class on
    a small calibration set, which reports a meaningless 0.5 AUROC even when
    the signal is real."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    fit, ev = [], []
    for cls in sorted(np.unique(labels)):
        ci = idx[labels == cls].copy()
        if len(ci) == 1:
            fit.append(ci)
            continue
        k = max(1, int(round(len(ci) * fit_frac)))
        if len(ci) - k == 0 and len(ci) > 1:
            k = len(ci) - 1
        rng.shuffle(ci)
        fit.append(ci[:k])
        ev.append(ci[k:])
    fit = np.concatenate(fit) if fit else np.array([], dtype=int)
    ev = np.concatenate(ev) if ev else np.array([], dtype=int)
    return fit, ev


def select_l_star(hid_by_sample: list[dict], labels: np.ndarray,
                  candidate_layers: list[int], fit_frac: float = 0.7
                  ) -> dict:
    """Per-layer linear-probe AUROC (probe fit on fit_frac of calibration,
    evaluated on the held-out remainder — no optimistic bias)."""
    n = len(hid_by_sample)
    rng = np.random.default_rng(0)
    fit_idx, eval_idx = _stratified_split(labels, rng, fit_frac)
    fit, eval_ = fit_idx, eval_idx

    out = {"per_layer": {}}
    for l in candidate_layers:
        X = np.array([hid_by_sample[i][l] for i in range(n)])
        y = (labels == 1).astype(int)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xs[fit], y[fit])
        if len(eval_) < 2:
            a = auroc(y, clf.predict_proba(Xs)[:, 1])
        else:
            a = auroc(y[eval_], clf.predict_proba(Xs[eval_])[:, 1])
        out["per_layer"][l] = float(a)
    best = max(out["per_layer"], key=out["per_layer"].get)
    out["best_layer"] = int(best)
    out["best_auroc"] = out["per_layer"][best]
    return out


def fit_probe(hid_by_sample: list[dict], labels: np.ndarray, layer: int,
              fit_frac: float = 0.7) -> tuple[dict, float, float]:
    """Fit the probe used at inference.

    Returns ``(probe_params, fit_auroc, eval_auroc)`` where ``eval_auroc`` is
    measured on the held-out fraction — the number that should be reported.
    Reporting ``fit_auroc`` (on training rows) would be an in-sample artifact.
    """
    n = len(hid_by_sample)
    X = np.array([hid_by_sample[i][layer] for i in range(n)])
    y = (labels == 1).astype(int)
    rng = np.random.default_rng(0)
    fit, ev = _stratified_split(labels, rng, fit_frac)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xs[fit], y[fit])
    proba = clf.predict_proba(Xs)[:, 1]
    fit_auroc = auroc(y[fit], proba[fit])
    eval_auroc = auroc(y[ev], proba[ev]) if len(ev) >= 2 else \
        auroc(y, proba)
    params = {"layer": int(layer),
              "coef": clf.coef_[0].tolist(),
              "bias": float(clf.intercept_[0]),
              "mean": scaler.mean_.tolist(),
              "std": scaler.scale_.tolist()}
    return params, float(fit_auroc), float(eval_auroc)


def probe_probs(hid: np.ndarray, probe: dict) -> np.ndarray:
    """P(injection) for one hidden vector under a saved probe."""
    z = (hid - np.array(probe["mean"])) / np.array(probe["std"])
    return 1.0 / (1.0 + np.exp(-(float(probe["bias"]) + float(z @ np.array(probe["coef"])))))


# ---------------------------------------------------------------------------
# Fusion: alpha grid search, threshold selection
# ---------------------------------------------------------------------------

def choose_alpha(r_scores: np.ndarray, p_scores: np.ndarray, labels: np.ndarray,
                 step: float = 0.05) -> dict:
    """Grid search alpha in S = a*z(R) + (1-a)*z(P) maximizing calibration AUROC.

    Standardization stats are computed on the same calibration set the probe
    was fit under (documented mild-leakage convention; the probe itself is
    evaluated on its held-out fraction).
    """
    zr, r_mu, r_sd = zscore(r_scores)
    zp, p_mu, p_sd = zscore(p_scores)
    best = (0.0, -1.0)
    curve = {}
    a = 0.0
    while a <= 1.0 + 1e-9:
        s = a * zr + (1 - a) * zp
        au = auroc(labels, s)
        curve[round(a, 2)] = float(au)
        if au > best[1]:
            best = (round(a, 2), au)
        a += step
    return {"alpha": float(best[0]), "auroc": float(best[1]), "curve": curve,
            "r_mu": r_mu, "r_sd": r_sd, "p_mu": p_mu, "p_sd": p_sd}


def choose_theta(scores: np.ndarray, labels: np.ndarray,
                 fpr_budget: float) -> dict:
    """Largest score threshold whose FPR <= fpr_budget; ties broken by max F1.

    fpr_budget for the meta 'per_spec' mode is target_fpr / n_type_specialists
    (union bound keeps overall FPR near the 5% target).
    """
    y = (labels == 1).astype(int)
    n_pos = max(1, int(y.sum()))
    n_neg = max(1, int((1 - y).sum()))
    order = np.argsort(-scores)
    best = None
    seen_theta = set()
    for i in range(len(scores) + 1):
        # threshold just above scores[order[i-1]] -> top (i-1) scores positive
        k = i - 1
        if k < 0:
            tp = fp = 0
            theta = float(scores.max() + 1e-9)
        else:
            pos = set(order[:k].tolist())
            tp = sum(1 for j in pos if y[j] == 1)
            fp = k - tp
            theta = float(scores[order[k - 1]]) + 1e-9
        if round(theta, 6) in seen_theta:
            continue
        seen_theta.add(round(theta, 6))
        fpr = fp / n_neg
        tpr = tp / n_pos
        if fpr <= fpr_budget + 1e-12:
            f1 = 2 * tp / (2 * tp + fp + (n_pos - tp)) if (2 * tp + fp + n_pos - tp) > 0 else 0.0
            cand = (theta, f1, tpr, fpr)
            if best is None or cand[1] > best[1]:
                best = cand
    if best is None:
        # every threshold violates the budget -> pick lowest-FPR operating point
        theta = float(scores.max() + 1e-9)
        best = (theta, 0.0, 0.0, 0.0)
    theta, f1, tpr, fpr = best
    return {"theta": theta, "f1_at_theta": float(f1),
            "tpr_at_theta": float(tpr), "fpr_at_theta": float(fpr),
            "fpr_budget": float(fpr_budget)}
