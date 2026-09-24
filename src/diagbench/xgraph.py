"""Cross-disease unified patient graph (MEGA27-10 discovery core).

One shared latent patient space across diseases: per-disease encoders map
heterogeneous clinical feature vectors into a common d_lat space; a single
kNN similarity graph is built over ALL patients (all diseases, transductive);
a multi-task GCN with one head per disease is trained on that graph.

Two falsifiable claims under test:
  C1 (benchmark): unified-graph diagnosis >= single-disease-graph diagnosis
      per disease (AUC, bootstrap CI over seeds).
  C2 (discovery): stable cross-disease "bridges" exist - cross-disease edges
      of the latent kNN graph whose removal measurably degrades the bridged
      diseases (ablation delta-AUC), recurring across seeds.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .eval import compute_metrics
from .graphs import knn_similarity_graph, normalize_adjacency
from .train import class_weights, set_seed



def meta_fingerprint(X: np.ndarray) -> np.ndarray:
    """Disease-agnostic 10-dim distributional fingerprint of a patient.

    Comparable across diseases because it does not depend on which native
    features were measured: per-patient moments/quantiles of the patient's
    own feature vector, column-standardized across the cohort given.
    """
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=1)
    sd = X.std(axis=1) + 1e-8
    z = (X - mu[:, None]) / sd[:, None]
    M = np.stack([
        mu, sd, X.min(axis=1), X.max(axis=1), np.median(X, axis=1),
        np.percentile(X, 25, axis=1), np.percentile(X, 75, axis=1),
        (z ** 3).mean(axis=1), (z ** 4).mean(axis=1), np.abs(z).mean(axis=1),
    ], axis=1)
    return ((M - M.mean(axis=0)) / (M.std(axis=0) + 1e-8)).astype(np.float32)

class Encoder(nn.Module):
    """Per-disease map from native features into the shared latent space."""

    def __init__(self, d_in: int, d_lat: int = 16, d_hid: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.Tanh(),
            nn.Linear(d_hid, d_hid), nn.Tanh(),
            nn.Linear(d_hid, d_lat), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskGCN(nn.Module):
    """Two-layer GCN trunk over the shared latent graph + per-disease heads."""

    def __init__(self, d_lat: int, n_diseases: int, d_hid: int = 16):
        super().__init__()
        self.w1 = nn.Linear(d_lat, d_hid)
        self.w2 = nn.Linear(d_hid, d_hid)
        self.heads = nn.ModuleList(nn.Linear(d_hid, 1)
                                   for _ in range(n_diseases))

    def forward(self, Z: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        H = torch.tanh(A_norm @ self.w1(Z))
        H = torch.tanh(A_norm @ self.w2(H))
        return torch.stack([h(H).squeeze(-1) for h in self.heads], dim=1)


def _standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return mu, sd


def train_encoders(datasets: list, idx_train: list, idx_val: list,
                   d_lat: int = 16, epochs: int = 300, lr: float = 1e-3,
                   patience: int = 60, seed: int = 0):
    """Phase A: supervised multi-task training of per-disease encoders.

    Returns (encoders, heads, Z_blocks) where Z_blocks[d] is the latent
    embedding of ALL patients of disease d under the frozen encoder.
    """
    set_seed(seed)
    encs = [Encoder(ds.X.shape[1], d_lat=d_lat) for ds in datasets]
    heads = nn.ModuleList(nn.Linear(d_lat, 1) for _ in datasets)
    params = [p for e in encs for p in e.parameters()] + list(heads.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    Xs = [torch.tensor(ds.X, dtype=torch.float32) for ds in datasets]
    ys = [torch.tensor(ds.y, dtype=torch.float32) for ds in datasets]
    trs = [torch.tensor(i, dtype=torch.long) for i in idx_train]
    vas = [torch.tensor(i, dtype=torch.long) for i in idx_val]
    best, best_state, stall = np.inf, None, 0
    for _ in range(epochs):
        tot = 0.0
        for d in range(len(datasets)):
            z = encs[d](Xs[d])
            logits = heads[d](z).squeeze(-1)
            w = class_weights(datasets[d].y[idx_train[d]])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[trs[d]], ys[d][trs[d]], pos_weight=w[1] / w[0])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        with torch.no_grad():
            vloss = 0.0
            for d in range(len(datasets)):
                z = encs[d](Xs[d])
                logits = heads[d](z).squeeze(-1)
                vloss += nn.functional.binary_cross_entropy_with_logits(
                    logits[vas[d]], ys[d][vas[d]]).item()
        if vloss < best - 1e-5:
            best, stall = vloss, 0
            best_state = ([{k: v.clone() for k, v in e.state_dict().items()}
                           for e in encs],
                          {k: v.clone() for k, v in heads.state_dict().items()})
        else:
            stall += 1
            if stall >= patience:
                break
    for e, st in zip(encs, best_state[0]):
        e.load_state_dict(st)
    heads.load_state_dict(best_state[1])
    with torch.no_grad():
        Z_blocks = [encs[d](Xs[d]).numpy() for d in range(len(datasets))]
    return encs, heads, Z_blocks


def build_unified_graph(Z_blocks: list, meta_blocks: list = None,
                        k: int = 10, keep_cross_disease: bool = True):
    """Union kNN graph over latent space AND distributional fingerprints.

    Latent kNN carries disease-specific similarity; meta kNN (disease-
    agnostic fingerprints) is what creates cross-disease bridges. With
    keep_cross_disease=False every cross-disease edge is removed (ablation).
    Node-feature matrix returned is [Z | meta]."""
    Z = np.concatenate(Z_blocks, axis=0).astype(np.float32)
    sizes = [len(z) for z in Z_blocks]
    block = np.concatenate([[d] * n for d, n in enumerate(sizes)])
    A = knn_similarity_graph(Z, k=k)
    if meta_blocks is not None:
        M = np.concatenate(meta_blocks, axis=0).astype(np.float32)
        A = np.maximum(A, knn_similarity_graph(M, k=k))
        X_node = np.concatenate([Z, M], axis=1)
    else:
        X_node = Z
    cross = block[:, None] != block[None, :]
    A_cross = A * cross
    if not keep_cross_disease:
        A = A * (~cross)
    return X_node, A, A_cross, block


def train_multitask_gcn(Z: np.ndarray, A: np.ndarray, block: np.ndarray,
                        ys: list, idx_train: list, idx_val: list,
                        epochs: int = 300, lr: float = 5e-3,
                        patience: int = 60, seed: int = 0):
    """Phase B: multi-task GCN over the unified (or ablated) graph."""
    set_seed(seed)
    n_dis = len(ys)
    model = MultiTaskGCN(Z.shape[1], n_dis)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    A_norm = torch.tensor(normalize_adjacency(A), dtype=torch.float32)
    Zt = torch.tensor(Z, dtype=torch.float32)
    sizes = [0] + list(np.cumsum([len(y) for y in ys]))
    y_all = torch.tensor(np.concatenate(ys), dtype=torch.float32)
    tr = [torch.tensor(idx_train[d] + sizes[d], dtype=torch.long)
          for d in range(n_dis)]
    va = [torch.tensor(idx_val[d] + sizes[d], dtype=torch.long)
          for d in range(n_dis)]
    best, best_state, stall = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        logits = model(Zt, A_norm)
        loss = 0.0
        for d in range(n_dis):
            w = class_weights(ys[d][idx_train[d]])
            loss = loss + nn.functional.binary_cross_entropy_with_logits(
                logits[tr[d], d], y_all[tr[d]], pos_weight=w[1] / w[0])
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vlogits = model(Zt, A_norm)
            vloss = sum(
                nn.functional.binary_cross_entropy_with_logits(
                    vlogits[va[d], d], y_all[va[d]]).item()
                for d in range(n_dis))
        if vloss < best - 1e-5:
            best, stall = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= patience:
                break
    model.load_state_dict(best_state)
    return model


def evaluate_per_disease(model, Z, A, block, ys, idx_test, seed: int = 0):
    """Test metrics per disease on the unified graph."""
    A_norm = torch.tensor(normalize_adjacency(A), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Z, dtype=torch.float32), A_norm).numpy()
    sizes = [0] + list(np.cumsum([len(y) for y in ys]))
    out = {}
    for d in range(len(ys)):
        te = idx_test[d] + sizes[d]
        p = 1.0 / (1.0 + np.exp(-logits[te, d]))
        out[d] = compute_metrics(ys[d][idx_test[d]], p)
    return out


def bridge_edges(A_cross: np.ndarray, block: np.ndarray, ys: list):
    """Named cross-disease bridges: label-annotated cross-disease edge pairs.

    Returns a list of dicts: diseases (i,j), endpoint labels, weight. These
    are the candidate discovery objects; recurrence across seeds + ablation
    delta-AUC make the claim falsifiable.
    """
    ii, jj = np.nonzero(np.triu(A_cross))
    edges = []
    sizes = [0] + list(np.cumsum([len(y) for y in ys]))
    for a, b in zip(ii.tolist(), jj.tolist()):
        da, db = int(block[a]), int(block[b])
        la = int(ys[da][a - sizes[da]])
        lb = int(ys[db][b - sizes[db]])
        edges.append({"diseases": sorted((da, db)), "labels": (la, lb),
                      "weight": float(A_cross[a, b])})
    return edges


def summarize_bridges(all_seed_edges: list, disease_names: list,
                      min_seed_recurrence: int = 4):
    """Aggregate bridge pairs across seeds; name the recurring ones."""
    from collections import Counter
    pair_count = Counter()
    label_sig = Counter()
    for edges in all_seed_edges:
        seen = set()
        for e in edges:
            key = tuple(e["diseases"])
            seen.add(key)
            label_sig[(key, tuple(e["labels"]))] += 1
        for key in seen:
            pair_count[key] += 1
    stable = []
    for (di, dj), c in sorted(pair_count.items()):
        if c >= min_seed_recurrence:
            sigs = {f"{la}-{lb}": n for ((di2, dj2), (la, lb)), n
                    in label_sig.items() if (di2, dj2) == (di, dj)}
            stable.append({
                "pair": f"{disease_names[di]}<->{disease_names[dj]}",
                "seed_recurrence": c,
                "endpoint_label_counts": sigs,
            })
    return stable
