#!/usr/bin/env python3
"""ROC figures from committed test probability dumps."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for d in ('malaria', 'pneumonia'):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for kind, label in (('cnn', 'CNN core'), ('gcn', 'RegionGCN')):
        p = os.path.join(ROOT, f'results/{d}/test_probs_{kind}.json')
        dat = json.load(open(p))
        probs = np.array(dat['probs']); labels = np.array(dat['labels'])
        if probs.ndim == 2: probs = probs[:, 1]
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, label=f'{label} (AUC {auc:.4f})')
    if d == 'pneumonia':
        dat = json.load(open(os.path.join(ROOT, 'results/pneumonia/test_probs_torchxrayvision.json')))
        probs = np.array(dat['probs']); labels = np.array(dat['labels'])
        if probs.ndim == 2: probs = probs[:, 1]
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, '--', label=f'TorchXRayVision (AUC {auc:.4f})')
    ax.plot([0, 1], [0, 1], ':', color='gray')
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(f'ROC on held-out test ({d})')
    out = os.path.join(ROOT, f'figures/{d}_roc_curves.png')
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print('wrote', out)
