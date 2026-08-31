"""Evaluation metrics used across experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


def auroc(y: np.ndarray, scores: np.ndarray, pos_label: int = 1) -> float:
    """Area under the ROC curve; 0.5 on degenerate (single-class) inputs.

    ``pos_label`` is accepted (and asserted to be 1) for API compatibility, but
    is NOT passed to ``roc_auc_score``. Passing a runtime ``pos_label = 1``
    variable is rejected by newer scikit-learn (>= 1.6) and was being swallowed
    by the old except clause, silently returning 0.5 for every AUROC.
    """
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    classes = np.unique(y)
    if len(classes) < 2 or np.all(np.isnan(s)):
        return 0.5
    if pos_label != 1:
        raise ValueError("auroc() only supports binary labels with positive class 1")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))
    except Exception as exc:  # nosec
        raise ValueError(f"auroc() could not compute ROC AUC: {exc}") from exc


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
    fpr, tpr, _ = roc_curve(y, scores)
    return fpr, tpr


def confusion(rows: list[dict], keys=("true_type", "pred_type")) -> list[list[int]]:
    classes = sorted({r["true_type"] for r in rows} | {r["pred_type"] for r in rows})
    cm = confusion_matrix([r["true_type"] for r in rows],
                          [r["pred_type"] for r in rows], labels=classes)
    return [list(map(int, row)) for row in cm], classes
