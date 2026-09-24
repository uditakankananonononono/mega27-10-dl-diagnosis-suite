"""Hermetic data-pipeline tests - no network, synthetic fixtures only."""
import numpy as np

from diagbench.data.base import (make_synthetic, standardize_train_test,
                                 stratified_split)


def test_synthetic_shape_and_labels():
    ds = make_synthetic(n=120, d=8, seed=1)
    assert ds.X.shape == (120, 8)
    assert ds.y.shape == (120,)
    assert set(np.unique(ds.y)) == {0, 1}
    assert len(ds.feature_names) == 8


def test_synthetic_separable_when_separation_high():
    # a linear probe must recover near-perfect labels on the training dist
    ds = make_synthetic(n=400, d=6, seed=2, separation=3.0)
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500).fit(ds.X, ds.y)
    assert clf.score(ds.X, ds.y) > 0.95


def test_stratified_split_preserves_ratio_and_disjoint():
    y = np.array([0] * 80 + [1] * 20)
    tr, te = stratified_split(y, 0.25, seed=3)
    assert len(set(tr) & set(te)) == 0
    assert len(tr) + len(te) == 100
    assert abs(y[te].mean() - 0.2) < 0.08  # stratified: ~20% positives


def test_standardize_no_leakage():
    rng = np.random.default_rng(0)
    Xtr = rng.normal(5.0, 2.0, size=(100, 4))
    Xte = rng.normal(50.0, 2.0, size=(20, 4))
    Ztr, Zte = standardize_train_test(Xtr, Xte)
    assert np.allclose(Ztr.mean(0), 0, atol=1e-6)
    assert np.allclose(Ztr.std(0), 1, atol=1e-6)
    # test block standardized with TRAIN stats -> not zero mean
    assert np.abs(Zte.mean()) > 5.0


def test_split_deterministic_per_seed():
    y = np.array([0, 1] * 50)
    a = stratified_split(y, 0.3, seed=9)
    b = stratified_split(y, 0.3, seed=9)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
