import numpy as np

from diagbench.eval import (bootstrap_ci, compute_metrics,
                            expected_calibration_error, summarize_runs)


def test_perfect_classifier_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.1, 0.9, 0.95])
    m = compute_metrics(y, p)
    assert m["roc_auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["brier"] < 0.01


def test_chance_classifier_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = rng.uniform(size=2000)
    m = compute_metrics(y, p)
    assert abs(m["roc_auc"] - 0.5) < 0.05


def test_ece_hand_computed():
    # one bin: 4 samples at p=0.7, half positive -> |0.5-0.7|=0.2
    y = np.array([0, 0, 1, 1])
    p = np.array([0.7, 0.7, 0.7, 0.7])
    assert abs(expected_calibration_error(y, p, n_bins=1) - 0.2) < 1e-9


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.3 + 0.4 * y + 0.1 * rng.normal(size=300), 0, 1)
    lo, hi = bootstrap_ci(y, p, n_boot=200, seed=1)
    auc = compute_metrics(y, p)["roc_auc"]
    assert lo <= auc <= hi


def test_summarize_runs():
    runs = [{"roc_auc": 0.9, "accuracy": 0.8}, {"roc_auc": 0.7, "accuracy": 0.9}]
    s = summarize_runs(runs)
    assert abs(s["roc_auc"]["mean"] - 0.8) < 1e-9
    assert abs(s["roc_auc"]["std"] - 0.1) < 1e-9
