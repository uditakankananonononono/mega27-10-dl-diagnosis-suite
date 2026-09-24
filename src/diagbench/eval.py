"""Metrics with bootstrap confidence intervals - the honesty layer."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score)


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE: mean over equal-width bins of |acc - conf|, sample-weighted."""
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(p)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if m.any():
            ece += (m.sum() / n) * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    yhat = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, yhat)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "brier": float(brier_score_loss(y_true, p)),
        "ece": expected_calibration_error(y_true, p),
    }
    return out


def bootstrap_ci(y_true: np.ndarray, p: np.ndarray, metric: str = "roc_auc",
                 n_boot: int = 1000, seed: int = 0) -> tuple:
    """Percentile bootstrap 95% CI for a metric."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p = np.asarray(p, dtype=float)
    stats = []
    n = len(p)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(compute_metrics(y_true[idx], p[idx])[metric])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_runs(runs: list) -> dict:
    """Aggregate per-seed metric dicts into mean +- std."""
    keys = [k for k in runs[0] if isinstance(runs[0][k], float)]
    return {k: {"mean": float(np.mean([r[k] for r in runs])),
                "std": float(np.std([r[k] for r in runs]))} for k in keys}
