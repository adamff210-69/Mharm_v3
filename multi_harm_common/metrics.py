"""Evaluation metrics used across experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve


def auroc(y: np.ndarray, scores: np.ndarray, pos_label: int = 1) -> float:
    """Area under the ROC curve; 0.5 on degenerate (single-class) inputs.

    NB: sklearn's ``roc_auc_score`` takes NO ``pos_label`` argument (that one
    belongs to ``roc_curve``/``precision_recall_curve``). v3.0 passed it anyway,
    every call raised TypeError, and the bare ``except: return 0.5`` swallowed
    it — so every AUROC in the whole project was a constant 0.5000 while still
    looking like a number. Binary labels are handled directly; anything else is
    reduced to one-vs-rest on ``pos_label``. Errors are never swallowed again.
    """
    y = np.asarray(y)
    s = np.asarray(scores, dtype=float)
    if s.shape[0] != y.shape[0] or s.shape[0] == 0:
        raise ValueError(f"auroc: label/score length mismatch {y.shape} vs {s.shape}")
    finite = ~np.isnan(s)
    if not finite.all():
        y, s = y[finite], s[finite]
    classes = np.unique(y)
    if len(classes) < 2:
        return 0.5
    if set(classes.tolist()) != {0, 1}:
        y = (y == pos_label).astype(int)
        if len(np.unique(y)) < 2:
            return 0.5
    from sklearn.metrics import roc_auc_score      # import error must be loud
    return float(roc_auc_score(y, s))


def auroc_selftest() -> dict:
    """Prove the metric actually computes, before any calibration depends on it.

    A metric that silently degrades to 0.5 corrupts head/layer selection, alpha
    and every table at once, and looks like a flat result rather than a bug —
    so every stage that reads AUROC should call this and abort on failure.
    """
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    perfect = auroc(y, np.array([-3., -2., -1., 0., 1., 2., 3., 4.]))
    reversed_ = auroc(y, -np.array([-3., -2., -1., 0., 1., 2., 3., 4.]))
    single = auroc(np.ones(6, dtype=int), np.arange(6.0))
    ok = (abs(perfect - 1.0) < 1e-12 and abs(reversed_ - 0.0) < 1e-12
          and single == 0.5)
    return {"ok": bool(ok), "perfect_separation": float(perfect),
            "inverted_separation": float(reversed_), "single_class": float(single)}


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


def roc_points(y: np.ndarray, scores: np.ndarray):
    fpr, tpr, _ = roc_curve(y, scores)
    return fpr, tpr


def confusion(rows: list[dict], keys=("true_type", "pred_type")) -> tuple:
    classes = sorted({r["true_type"] for r in rows} | {r["pred_type"] for r in rows})
    cm = confusion_matrix([r["true_type"] for r in rows],
                          [r["pred_type"] for r in rows], labels=classes)
    return [list(map(int, row)) for row in cm], classes
