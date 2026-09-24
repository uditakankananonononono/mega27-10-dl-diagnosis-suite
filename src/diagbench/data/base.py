"""Dataset registry and splitting utilities for the diagnosis benchmark suite.

Every loader returns a TabularDataset of REAL clinical data. Tests never call
these loaders over the network; they use make_synthetic() instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import urllib.request

import numpy as np

CACHE_DIR = os.environ.get(
    "DIAGBENCH_CACHE",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data_cache"),
)


class DataUnavailable(RuntimeError):
    """Raised when a live download cannot be completed (CI / offline)."""


@dataclass
class TabularDataset:
    name: str
    X: np.ndarray  # (n, d) float32 raw features
    y: np.ndarray  # (n,) int64 binary labels, 1 = disease/positive
    feature_names: list
    positive_label: str
    source_url: str
    citation: str = ""

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)
        assert self.X.ndim == 2 and self.y.ndim == 1
        assert len(self.X) == len(self.y), "X and y length mismatch"
        assert set(np.unique(self.y)) <= {0, 1}, "labels must be binary"
        assert self.X.shape[1] == len(self.feature_names)


def fetch_text(url: str, cache_name: str, timeout: int = 60) -> str:
    """Download `url` to the local cache and return its text.

    Network access happens ONLY here; every parsing step is pure and testable.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, cache_name)
    if not os.path.exists(path):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
        except Exception as exc:  # pragma: no cover - network dependent
            raise DataUnavailable(f"could not download {url}: {exc}") from exc
        with open(path, "wb") as fh:
            fh.write(data)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def make_synthetic(n: int = 200, d: int = 12, seed: int = 0,
                   separation: float = 2.0) -> TabularDataset:
    """Gaussian two-class problem used by the hermetic test-suite."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    w = w / np.linalg.norm(w)
    X = rng.normal(size=(n, d)).astype(np.float32)
    logit = separation * (X @ w) + 0.2 * rng.normal(size=n)
    y = (logit > 0).astype(np.int64)
    return TabularDataset(
        name="synthetic", X=X, y=y,
        feature_names=[f"f{i}" for i in range(d)],
        positive_label="positive", source_url="generated",
    )


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    """Z-score with train statistics; never leak test statistics."""
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


def stratified_split(y: np.ndarray, test_frac: float, seed: int):
    """Stratified train/test indices preserving class proportions."""
    rng = np.random.default_rng(seed)
    idx_train, idx_test = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(test_frac * len(idx))))
        idx_test.extend(idx[:n_test].tolist())
        idx_train.extend(idx[n_test:].tolist())
    return np.array(sorted(idx_train)), np.array(sorted(idx_test))
