"""Confident-learning label-noise census, implemented from scratch
(Northcutt, Jiang & Chuang 2021, JAIR - the confident joint estimator).

Given out-of-fold predicted probabilities P[n,2] and given labels y, we
estimate the confident joint C[i,j] = count of samples labelled i whose
confident prediction is class j, with per-class thresholds t_j = mean
predicted prob of class j among samples labelled j. Off-diagonal entries are
flagged as likely label errors. Every flagged sample is reported by stable ID.
"""
from __future__ import annotations

import numpy as np


def confident_thresholds(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    t = np.zeros(probs.shape[1])
    for j in range(probs.shape[1]):
        mask = labels == j
        t[j] = probs[mask, j].mean() if mask.any() else 0.5
    return t


def confident_joint(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    t = confident_thresholds(probs, labels)
    K = probs.shape[1]
    C = np.zeros((K, K), dtype=int)
    confident_pred = np.where(probs >= t[None, :], probs, -np.inf).argmax(1)
    for i in range(K):
        for j in range(K):
            C[i, j] = int(((labels == i) & (confident_pred == j)).sum())
    return C


def flag_label_errors(probs: np.ndarray, labels: np.ndarray,
                      margin: float = 0.0) -> np.ndarray:
    """Boolean mask of suspected label errors: confident prediction disagrees
    with the given label (prob of the other class above its threshold)."""
    t = confident_thresholds(probs, labels)
    other = 1 - labels
    confident_other = (probs[np.arange(len(labels)), other]
                       >= t[other] + margin)
    return confident_other


def noise_summary(probs: np.ndarray, labels: np.ndarray) -> dict:
    C = confident_joint(probs, labels)
    n = len(labels)
    off = int(C[0, 1] + C[1, 0])
    return {
        "confident_joint": C.tolist(),
        "estimated_label_errors": off,
        "estimated_noise_rate": off / n,
        "per_class_error": {
            "class0_labelled_pred1": int(C[0, 1]),
            "class1_labelled_pred0": int(C[1, 0]),
        },
        "thresholds": confident_thresholds(probs, labels).tolist(),
        "n": n,
    }
