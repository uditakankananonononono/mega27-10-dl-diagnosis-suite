"""Patient-similarity graph construction for the GNN models."""
from __future__ import annotations

import numpy as np


def knn_similarity_graph(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Symmetric kNN graph on standardized features, cosine-free version.

    A_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)) for j in kNN(i), symmetrized
    by union (max). sigma is the median kNN distance (self-tuning, Zelnik-Manor
    & Perona 2004 style). Returns dense float32 adjacency, zero diagonal.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    k = max(1, min(k, n - 1))
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    knn_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    knn_d2 = np.take_along_axis(d2, knn_idx, axis=1)
    sigma2 = np.median(knn_d2) + 1e-8
    W = np.zeros((n, n), dtype=np.float32)
    np.put_along_axis(W, knn_idx, np.exp(-knn_d2 / (2.0 * sigma2)), axis=1)
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 0.0)
    return W


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Symmetric normalization D^{-1/2} (A + I) D^{-1/2} (Kipf & Welling)."""
    A = np.asarray(A, dtype=np.float32)
    A_hat = A + np.eye(A.shape[0], dtype=np.float32)
    deg = A_hat.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    return (d_inv_sqrt[:, None] * A_hat) * d_inv_sqrt[None, :]
