"""Ranking metrics, threshold metrics and threshold selection."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _prepare(y, s) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y and scores must have the same shape")
    if y.size == 0:
        raise ValueError("empty input")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return y, s


def ranking_metrics(y, s) -> Dict[str, float]:
    y, s = _prepare(y, s)
    if len(np.unique(y)) < 2:
        raise ValueError("both classes are required to compute ROC-AUC and PR-AUC")
    return {"roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
            "prevalence": float(y.mean())}


def threshold_metrics(y, s, threshold: float) -> Dict[str, float]:
    """Rows with score >= threshold are flagged."""
    y, s = _prepare(y, s)
    pred = s >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1,
            "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "flagged_rate": float(pred.mean())}


def select_threshold_max_f1(y, s) -> float:
    y, s = _prepare(y, s)
    if len(np.unique(y)) < 2:
        raise ValueError("both classes are required to select a threshold")
    p, r, thr = precision_recall_curve(y, s)
    p, r = p[:-1], r[:-1]
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thr[int(np.argmax(f1))])


def select_threshold_fpr_budget(y, s, max_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on (y, s) does not exceed ``max_fpr``."""
    y, s = _prepare(y, s)
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be in [0, 1]")
    neg = np.sort(s[y == 0])[::-1]
    if neg.size == 0:
        raise ValueError("no negative samples to calibrate the threshold")
    k = int(np.floor(max_fpr * neg.size))
    if k >= neg.size:
        return float(neg[-1])
    return float(np.nextafter(neg[k], np.inf))
