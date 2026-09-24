"""Training loops with early stopping, class weighting and seeded determinism."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .graphs import knn_similarity_graph, normalize_adjacency


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y_train: np.ndarray) -> torch.Tensor:
    n0 = float((y_train == 0).sum())
    n1 = float((y_train == 1).sum())
    return torch.tensor([1.0 / max(n0, 1.0), 1.0 / max(n1, 1.0)],
                        dtype=torch.float32)


def _early_stop_state(model: nn.Module):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_tabular(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  epochs: int = 400, lr: float = 1e-3, wd: float = 1e-4,
                  patience: int = 60, seed: int = 0) -> nn.Module:
    """Minibatch training for MLP / CNN1D with early stopping on val loss."""
    set_seed(seed)
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xva = torch.tensor(X_val, dtype=torch.float32)
    yva = torch.tensor(y_val, dtype=torch.float32)
    w = class_weights(y_train)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_loss, best_state, stall = np.inf, _early_stop_state(model), 0
    n = len(Xtr)
    bs = min(64, n)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = model(Xtr[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, ytr[idx], pos_weight=w[1] / w[0])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = nn.functional.binary_cross_entropy_with_logits(
                model(Xva), yva).item()
        if vloss < best_loss - 1e-5:
            best_loss, best_state, stall = vloss, _early_stop_state(model), 0
        else:
            stall += 1
            if stall >= patience:
                break
    model.load_state_dict(best_state)
    return model


def train_graph(model: nn.Module, X: np.ndarray, y: np.ndarray,
                idx_train: np.ndarray, idx_val: np.ndarray, k: int = 10,
                epochs: int = 500, lr: float = 5e-3, wd: float = 5e-4,
                patience: int = 80, seed: int = 0,
                arch: str = "gcn") -> tuple:
    """Full-batch transductive training for GCN/GAT on the patient graph.

    The graph is built from ALL nodes' features (unlabeled transduction);
    labels are used only on idx_train. Returns (model, A_used).
    """
    set_seed(seed)
    A = knn_similarity_graph(X, k=k)
    if arch == "gcn":
        A_in = torch.tensor(normalize_adjacency(A), dtype=torch.float32)
    else:  # gat attends over the raw graph incl. self-loops
        A_in = torch.tensor((A > 0).astype(np.float32) + np.eye(len(A)),
                            dtype=torch.float32)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    tr = torch.tensor(idx_train, dtype=torch.long)
    va = torch.tensor(idx_val, dtype=torch.long)
    w = class_weights(y[idx_train])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_loss, best_state, stall = np.inf, _early_stop_state(model), 0
    for _ in range(epochs):
        model.train()
        logits = model(Xt, A_in)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits[tr], yt[tr], pos_weight=w[1] / w[0])
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = nn.functional.binary_cross_entropy_with_logits(
                model(Xt, A_in)[va], yt[va]).item()
        if vloss < best_loss - 1e-5:
            best_loss, best_state, stall = vloss, _early_stop_state(model), 0
        else:
            stall += 1
            if stall >= patience:
                break
    model.load_state_dict(best_state)
    return model, A_in
