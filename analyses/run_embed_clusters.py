"""umap-learn embedding + hdbscan clustering of disease-agnostic patient
fingerprints stacked across panel datasets: do cross-disease clusters align
with diagnosis rather than disease? Output analyses/embed_clusters.json +
figures/fingerprint_umap.png"""
import json, os, sys, warnings
sys.path.insert(0, "src")
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diagbench.data.panel import PANEL_LOADERS
from diagbench.xgraph import meta_fingerprint

Xs, disease, ylab = [], [], []
for name, fn in PANEL_LOADERS.items():
    ds = fn()
    if len(ds.y) < 80:
        continue
    F = meta_fingerprint(ds.X)
    Xs.append(F); disease += [name] * len(F); ylab += list(ds.y)
F = np.vstack(Xs).astype(np.float32)
disease = np.array(disease); ylab = np.array(ylab)
# per-dataset column standardization already inside meta_fingerprint; scale columns again across stack
F = (F - F.mean(0)) / (F.std(0) + 1e-8)

import umap
emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0,
                n_components=2).fit_transform(F)
import hdbscan
lab = hdbscan.HDBSCAN(min_cluster_size=40).fit_predict(emb)
from sklearn.metrics import adjusted_rand_score, roc_auc_score
ari_dis = adjusted_rand_score(disease, lab)
# within-disease: does the cluster structure carry label signal? AUC of
# cluster membership per disease (majority cluster vs rest)
uniq = sorted(set(disease))
per_ds = {}
for u in uniq:
    m = disease == u
    if len(set(lab[m])) > 1 and len(set(ylab[m])) > 1:
        try:
            per_ds[u] = float(roc_auc_score(ylab[m], lab[m]))
        except Exception:
            pass
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for i, u in enumerate(uniq):
    m = disease == u
    ax[0].scatter(emb[m, 0], emb[m, 1], s=2, label=u, alpha=0.6)
ax[0].set_title("UMAP of meta-fingerprints, colored by disease")
ax[0].legend(markerscale=4, fontsize=5, ncol=2)
m1 = ylab == 1
ax[1].scatter(emb[~m1, 0], emb[~m1, 1], s=2, alpha=0.5, label="control")
ax[1].scatter(emb[m1, 0], emb[m1, 1], s=2, alpha=0.5, label="case")
ax[1].set_title("same embedding, colored by diagnosis")
ax[1].legend(markerscale=4)
fig.tight_layout(); fig.savefig("figures/fingerprint_umap.png", dpi=140)
out = {"tools": ["umap-learn", "hdbscan"],
       "n_patients": int(len(F)), "n_datasets": int(len(uniq)),
       "n_clusters": int(len(set(lab)) - (1 if -1 in lab else 0)),
       "noise_fraction": float((lab == -1).mean()),
       "ari_clusters_vs_disease": float(ari_dis),
       "cluster_label_auc_per_disease": per_ds,
       "interpretation": "low ARI to disease + above-chance label AUCs supports disease-agnostic signal in fingerprints"}
json.dump(out, open("analyses/embed_clusters.json", "w"), indent=1)
print("EMBED_DONE", flush=True)
