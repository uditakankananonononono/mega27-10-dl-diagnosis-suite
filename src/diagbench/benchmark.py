"""End-to-end benchmark runner: datasets x models x seeds, fully recorded."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data.base import TabularDataset, standardize_train_test, stratified_split
from .eval import bootstrap_ci, compute_metrics, summarize_runs
from .graphs import knn_similarity_graph
from .models.cnn1d import CNN1D
from .models.gnn import GAT, GCN
from .models.mlp import MLP
from .train import set_seed, train_graph, train_tabular

DEEP_MODELS = ("mlp", "cnn1d", "gcn", "gat")
SKLEARN_BASELINES = ("logreg", "random_forest")


def _val_split(idx_train: np.ndarray, y: np.ndarray, seed: int,
               val_frac: float = 0.2):
    """Carve a stratified validation set out of the training indices."""
    rng = np.random.default_rng(seed + 777)
    tr, va = [], []
    for cls in (0, 1):
        idx = idx_train[y[idx_train] == cls].copy()
        rng.shuffle(idx)
        n_va = max(1, int(round(val_frac * len(idx))))
        va.extend(idx[:n_va].tolist())
        tr.extend(idx[n_va:].tolist())
    return np.array(sorted(tr)), np.array(sorted(va))


def run_one_seed(ds: TabularDataset, model_name: str, seed: int,
                 test_frac: float = 0.25, k: int = 10) -> dict:
    """Train and evaluate one model on one seeded split. Pure CPU, small."""
    set_seed(seed)
    idx_train, idx_test = stratified_split(ds.y, test_frac, seed)

    if model_name in SKLEARN_BASELINES:
        if model_name == "logreg":
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000,
                                                   class_weight="balanced"))
        else:
            clf = RandomForestClassifier(n_estimators=300,
                                         class_weight="balanced_subsample",
                                         random_state=seed, n_jobs=2)
        clf.fit(ds.X[idx_train], ds.y[idx_train])
        p = clf.predict_proba(ds.X[idx_test])[:, 1]
        m = compute_metrics(ds.y[idx_test], p)
        m["roc_auc_ci95"] = bootstrap_ci(ds.y[idx_test], p, seed=seed)
        return m

    Xtr, Xte = standardize_train_test(ds.X[idx_train], ds.X[idx_test])
    itr, iva = _val_split(idx_train, ds.y, seed)
    # map global train indices to positions inside the standardized block
    pos = {g: i for i, g in enumerate(idx_train.tolist())}
    ptr = np.array([pos[g] for g in itr])
    pva = np.array([pos[g] for g in iva])
    d = ds.X.shape[1]

    if model_name == "mlp":
        model = MLP(d)
        model = train_tabular(model, Xtr[ptr], ds.y[itr], Xtr[pva],
                              ds.y[iva], seed=seed)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(Xte, dtype=torch.float32))
        p = torch.sigmoid(logits).numpy()

    elif model_name == "cnn1d":
        model = CNN1D(d)
        model = train_tabular(model, Xtr[ptr], ds.y[itr], Xtr[pva],
                              ds.y[iva], seed=seed)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(Xte, dtype=torch.float32))
        p = torch.sigmoid(logits).numpy()

    elif model_name in ("gcn", "gat"):
        # transductive: graph spans train+test nodes, labels only on train
        Xall = np.vstack([Xtr, Xte])
        yall = np.concatenate([ds.y[idx_train], ds.y[idx_test]])
        ntr = len(Xtr)
        arch = model_name
        model = GCN(d) if arch == "gcn" else GAT(d)
        # validation: hold out part of train inside the graph
        rng = np.random.default_rng(seed + 555)
        perm = rng.permutation(ntr)
        n_va = max(2, int(round(0.2 * ntr)))
        va_in, tr_in = perm[:n_va], perm[n_va:]
        model, A_in = train_graph(model, Xall, yall, tr_in, va_in, k=k,
                                  seed=seed, arch=arch)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(Xall, dtype=torch.float32), A_in)
        p = torch.sigmoid(logits[ntr:]).numpy()
    else:
        raise KeyError(model_name)

    m = compute_metrics(ds.y[idx_test], p)
    m["roc_auc_ci95"] = bootstrap_ci(ds.y[idx_test], p, seed=seed)
    return m


def run_benchmark(ds: TabularDataset, models=("mlp", "cnn1d", "gcn", "gat",
                                              "logreg", "random_forest"),
                  seeds=(0, 1, 2, 3, 4), k: int = 10) -> dict:
    t0 = time.time()
    out = {"dataset": ds.name, "n": int(ds.X.shape[0]),
           "d": int(ds.X.shape[1]), "source": ds.source_url,
           "citation": ds.citation, "seeds": list(seeds), "models": {}}
    for model_name in models:
        runs = [run_one_seed(ds, model_name, s, k=k) for s in seeds]
        out["models"][model_name] = {
            "per_seed": runs,
            "summary": summarize_runs(runs),
        }
    out["wall_seconds"] = round(time.time() - t0, 2)
    return out


def save_results(result: dict, path: str):
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
