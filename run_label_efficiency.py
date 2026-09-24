"""Label efficiency: transductive GCN at L% labels vs MLP at 100% labels.

Claim under test (falsifiable): on small clinical cohorts, a patient-graph
GCN trained on a FRACTION of labels matches or beats an MLP trained on ALL
labels. This is where graph diagnosis could set a real record.
"""
import json, os, sys, time
sys.path.insert(0, "src")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch; torch.set_num_threads(2)
from diagbench.data.clinical import load_dataset
from diagbench.data.base import standardize_train_test, stratified_split
from diagbench.eval import compute_metrics
from diagbench.models.mlp import MLP
from diagbench.models.gnn import GCN
from diagbench.train import train_tabular, train_graph, set_seed

DATASETS = ("wdbc", "cleveland", "pima", "parkinsons")
FRACS = (0.10, 0.25, 0.50, 1.00)
SEEDS = (0, 1, 2, 3, 4)

out = {}
for name in DATASETS:
    ds = load_dataset(name)
    out[name] = {"gcn": {f: [] for f in FRACS}, "mlp": {f: [] for f in FRACS}}
    for seed in SEEDS:
        idx_train, idx_test = stratified_split(ds.y, 0.25, seed)
        Xtr, Xte = standardize_train_test(ds.X[idx_train], ds.X[idx_test])
        for frac in FRACS:
            set_seed(seed * 100 + int(frac * 100))
            rng = np.random.default_rng(seed * 7 + int(frac * 100))
            n_lab = max(4, int(round(frac * len(idx_train))))
            sub = rng.choice(len(idx_train), size=n_lab, replace=False)
            sub.sort()
            # MLP on the labeled subset (tabular)
            mlp = MLP(ds.X.shape[1])
            mlp = train_tabular(mlp, Xtr[sub], ds.y[idx_train[sub]],
                                Xtr[sub], ds.y[idx_train[sub]],
                                epochs=300, patience=40, seed=seed)
            mlp.eval()
            with torch.no_grad():
                p = torch.sigmoid(mlp(torch.tensor(Xte, dtype=torch.float32))).numpy()
            out[name]["mlp"][frac].append(compute_metrics(ds.y[idx_test], p)["roc_auc"])
            # GCN transductive: graph over train+test, labels on subset
            ntr = len(idx_train)
            Xall = np.concatenate([Xtr, Xte], axis=0)
            yall = np.concatenate([ds.y[idx_train], ds.y[idx_test]])
            gcn = GCN(ds.X.shape[1])
            va = sub[:max(2, n_lab // 5)]
            tr = sub[max(2, n_lab // 5):]
            gcn, _ = train_graph(gcn, Xall, yall, tr, va, k=10,
                                 epochs=400, patience=50, seed=seed)
            gcn.eval()
            from diagbench.graphs import knn_similarity_graph, normalize_adjacency
            A = torch.tensor(normalize_adjacency(knn_similarity_graph(Xall, k=10)), dtype=torch.float32)
            with torch.no_grad():
                logits = gcn(torch.tensor(Xall, dtype=torch.float32), A).numpy()
            p = 1 / (1 + np.exp(-logits[ntr:]))
            out[name]["gcn"][frac].append(compute_metrics(ds.y[idx_test], p)["roc_auc"])
        print(f"{name} seed {seed} done", flush=True)
    json.dump(out, open("results/label_efficiency.json", "w"), indent=1)
    print(f"SAVED {name}", flush=True)
print("LABEL_EFF_DONE", flush=True)
for name in out:
    for f in FRACS:
        g = np.mean(out[name]["gcn"][f]); m = np.mean(out[name]["mlp"][f])
        print(f"{name:11s} labels {int(f*100):3d}%  GCN {g:.4f}  MLP {m:.4f}", flush=True)
