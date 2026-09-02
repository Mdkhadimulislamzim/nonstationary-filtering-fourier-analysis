"""
metrics.py -- 

Required: classification accuracy, macro-F1, confusion matrix, class-wise
sensitivity/recall, training and inference time.  Supplementary (only where
a clean reference exists, i.e. the Section 5.5 experiment): PSNR and SSIM.

Also provided: the paired McNemar test.  The OCTMNIST test set holds 1,000
images, so a one-point accuracy difference is within sampling noise; the
three approaches are evaluated on exactly the same test images, which makes
a paired test both available and necessary.
"""

from __future__ import annotations

import numpy as np

__all__ = ["classification_report", "mcnemar", "psnr", "ssim", "summarise_runs"]


def classification_report(y_true, y_pred, logits=None, n_classes=None) -> dict:
    from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                                 recall_score, roc_auc_score)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    K = n_classes or int(max(y_true.max(), y_pred.max()) + 1)
    labels = list(range(K))

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels,
                                   zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "per_class_recall": recall_score(y_true, y_pred, average=None, labels=labels,
                                         zero_division=0),
    }
    if logits is not None:
        p = _softmax(np.asarray(logits))
        try:
            out["auc"] = float(roc_auc_score(y_true, p if K > 2 else p[:, 1],
                                             multi_class="ovr", average="macro"))
        except ValueError:
            out["auc"] = float("nan")
    return out


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def mcnemar(y_true, pred_a, pred_b, exact: bool = True) -> dict:
    """Paired comparison of two classifiers on identical test items.

    b = A right / B wrong, c = A wrong / B right.  Exact binomial test when
    b + c is small, chi-square with continuity correction otherwise.
    """
    from scipy import stats
    y_true = np.asarray(y_true).reshape(-1)
    a = np.asarray(pred_a).reshape(-1) == y_true
    b_ = np.asarray(pred_b).reshape(-1) == y_true
    b = int(np.sum(a & ~b_))
    c = int(np.sum(~a & b_))
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "test": "degenerate"}
    if exact and n < 40:
        p = float(stats.binomtest(b, n, 0.5).pvalue)
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": p,
                "test": "exact binomial"}
    stat = (abs(b - c) - 1.0) ** 2 / n
    return {"b": b, "c": c, "statistic": float(stat),
            "p_value": float(stats.chi2.sf(stat, 1)), "test": "chi2 (corrected)"}


def holm(p_values: dict) -> dict:
    """Holm step-down correction over a dict {name: p}."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


# ---------------------------------------------------------------------------
# Supplementary image-quality metrics (Section 5.5 only)
# ---------------------------------------------------------------------------

def psnr(clean, restored, data_range: float = 1.0) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    restored = np.asarray(restored, dtype=np.float64)
    mse = np.mean((clean - restored) ** 2)
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(clean, restored, data_range: float = 1.0) -> float:
    from skimage.metrics import structural_similarity
    clean = np.asarray(clean, dtype=np.float64)
    restored = np.asarray(restored, dtype=np.float64)
    if clean.ndim == 2:
        return float(structural_similarity(clean, restored, data_range=data_range))
    vals = [structural_similarity(c, r, data_range=data_range,
                                  channel_axis=-1 if c.ndim == 3 else None)
            for c, r in zip(clean, restored)]
    return float(np.mean(vals))


def summarise_runs(reports: list) -> dict:
    """Mean and standard deviation over seeds for the scalar metrics."""
    keys = ["accuracy", "macro_f1", "auc"]
    out = {}
    for k in keys:
        vals = [r[k] for r in reports if k in r and np.isfinite(r[k])]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    out["n_seeds"] = len(reports)
    return out
