"""Training/eval engine: deterministic seeds, early stopping, full metrics."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader


def train_model(model, train_ds, val_ds, *, epochs: int, batch: int = 64,
                lr: float = 1e-3, seed: int = 0, threads: int = 2,
                patience: int = 3, log_every: int = 50,
                sample_weight: np.ndarray | None = None):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weights = None
    if sample_weight is not None:
        weights = torch.tensor(sample_weight, dtype=torch.float32)
    best = {"val_loss": float("inf"), "state": None, "epoch": -1}
    history = []
    bad = 0
    for ep in range(epochs):
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            if weights is None:
                loss = nn.functional.cross_entropy(logits, yb)
            else:
                loss = nn.functional.cross_entropy(
                    logits, yb, reduction="none")
                loss = (loss * weights[yb]).mean()   # per-class weights
            loss.backward()
            opt.step()
            running += loss.item(); nb += 1
        vl = evaluate_loss(model, val_ds, batch, threads)
        history.append({"epoch": ep, "train_loss": running / max(nb, 1),
                        "val_loss": vl, "secs": round(time.time() - t0, 1)})
        print(f"  epoch {ep}: train_loss={running/max(nb,1):.4f} val_loss={vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vl < best["val_loss"] - 1e-4:
            best = {"val_loss": vl, "epoch": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, history, best


@torch.no_grad()
def evaluate_loss(model, ds, batch: int, threads: int) -> float:
    torch.set_num_threads(threads)
    model.eval()
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    tot, n = 0.0, 0
    for xb, yb in loader:
        logits = model(xb)
        tot += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        n += len(yb)
    return tot / max(n, 1)


@torch.no_grad()
def predict_proba(model, ds, batch: int = 128, threads: int = 2) -> np.ndarray:
    torch.set_num_threads(threads)
    model.eval()
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    probs, labels = [], []
    for xb, yb in loader:
        probs.append(torch.softmax(model(xb), dim=1).numpy())
        labels.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def full_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    pred = probs.argmax(1)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(labels, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs[:, 1])),
        "confusion_matrix": cm.tolist(),
        "n": int(len(labels)),
    }


def save_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
