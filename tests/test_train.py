"""End-to-end hermetic training smoke tests on synthetic data."""
import numpy as np
import torch

from diagbench.data.base import (make_synthetic, standardize_train_test,
                                 stratified_split)
from diagbench.eval import compute_metrics
from diagbench.models.cnn1d import CNN1D
from diagbench.models.gnn import GCN
from diagbench.models.mlp import MLP
from diagbench.train import train_graph, train_tabular


def _split(ds, seed=0):
    tr, te = stratified_split(ds.y, 0.25, seed)
    Xtr, Xte = standardize_train_test(ds.X[tr], ds.X[te])
    return tr, te, Xtr, Xte


def test_train_mlp_learns_synthetic():
    ds = make_synthetic(n=300, d=10, seed=5, separation=2.5)
    tr, te, Xtr, Xte = _split(ds)
    model = MLP(10, hidden=(32,))
    n_va = max(4, len(tr) // 5)
    model = train_tabular(model, Xtr[:-n_va], ds.y[tr][:-n_va],
                          Xtr[-n_va:], ds.y[tr][-n_va:], epochs=150, seed=0)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(Xte))).numpy()
    assert compute_metrics(ds.y[te], p)["roc_auc"] > 0.9


def test_train_cnn_learns_synthetic():
    ds = make_synthetic(n=300, d=10, seed=6, separation=2.5)
    tr, te, Xtr, Xte = _split(ds)
    model = CNN1D(10, channels=(16, 8))
    n_va = max(4, len(tr) // 5)
    model = train_tabular(model, Xtr[:-n_va], ds.y[tr][:-n_va],
                          Xtr[-n_va:], ds.y[tr][-n_va:], epochs=400, lr=2e-3, seed=0)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(Xte))).numpy()
    assert compute_metrics(ds.y[te], p)["roc_auc"] > 0.85


def test_train_gcn_learns_clustered_data():
    rng = np.random.default_rng(7)
    # cluster-conditional labels: graph structure carries the signal
    X = np.vstack([rng.normal(0, 0.4, (60, 6)), rng.normal(4, 0.4, (60, 6))])
    y = np.array([0] * 60 + [1] * 60)
    X = (X - X.mean(0)) / X.std(0)
    idx = rng.permutation(120)
    tr, va = idx[:80], idx[80:]
    model = GCN(6, hidden=16)
    model, A = train_graph(model, X.astype(np.float32), y, tr, va[:20],
                           k=8, epochs=300, seed=0)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32), A))
    acc = ((p.numpy() > 0.5) == y).mean()
    assert acc > 0.9
