"""Evaluation metrics used across experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


def auroc(y: np.ndarray, scores: np.ndarray, pos_label: int = 1) -> float:
    """Area under the ROC curve.

    Returns NaN (not 0.5) when AUROC is undefined — a single-class slice
    or a sklearn error. Callers that previously saw a wall of 0.5000 were
    evaluating injected-only subsets (all labels == 1). Use
    :func:`type_vs_clean_ids` / :func:`type_vs_clean_mask` for per-type
    *detection* AUROC.
    """
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    y_bin = (y == pos_label).astype(int)
    if y_bin.size == 0 or np.all(np.isnan(s)) or len(np.unique(y_bin)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_bin, s))
    except TypeError:
        return float(roc_auc_score(y, s, pos_label=pos_label))


def type_vs_clean_ids(df, split: str, attack_type: str) -> list:
    """Ids for detection AUROC of one attack type on ``split``.

    Clean rows are stored as ``attack_type='clean'``. Filtering to the
    attack type alone is an all-positive set, so AUROC is undefined.
    This returns injected-of-type **plus** clean negatives from the same
    split.
    """
    inj = df[(df["split"] == split) & (df["attack_type"] == attack_type)]["id"]
    cln = df[(df["split"] == split) & (df["label"] == 0)]["id"]
    return inj.tolist() + cln.tolist()


def type_vs_clean_mask(types, labels, attack_type) -> np.ndarray:
    """Boolean mask: this attack type's injected rows + all clean rows."""
    types = np.asarray(types)
    labels = np.asarray(labels)
    return (types == attack_type) | (labels == 0)


def tpr_fpr(y: np.ndarray, scores: np.ndarray, theta: float) -> tuple[float, float]:
    y = np.asarray(y)
    pred = (np.asarray(scores, dtype=float) > theta).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return tpr, fpr


def asr(y: np.ndarray, scores: np.ndarray, theta: float) -> float:
    """Attack success rate = fraction of injected samples that evade detection
    (= 1 - TPR). v3 success criteria are stated in ASR terms."""
    tpr, _ = tpr_fpr(y, scores, theta)
    return 1.0 - tpr


def f1(y: np.ndarray, scores: np.ndarray, theta: float) -> float:
    y = np.asarray(y)
    pred = (np.asarray(scores, dtype=float) > theta).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / max(1e-12, prec + rec)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def spread(vals: list[float]) -> float:
    """Max-min spread across per-attack-type AUROCs (v3 §4.8 column)."""
    return max(vals) - min(vals) if vals else 0.0


def roc_points(y: np.ndarray, scores: np.ndarray):
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr


def confusion(rows: list[dict], keys=("true_type", "pred_type")) -> list[list[int]]:
    classes = sorted({r["true_type"] for r in rows} | {r["pred_type"] for r in rows})
    cm = confusion_matrix([r["true_type"] for r in rows],
                          [r["pred_type"] for r in rows], labels=classes)
    return [list(map(int, row)) for row in cm], classes
