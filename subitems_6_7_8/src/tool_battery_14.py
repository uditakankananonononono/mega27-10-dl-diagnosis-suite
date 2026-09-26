"""Battery 14: torchvision pretrained ResNet18 transfer baseline for neuro,
hdbscan cluster-vs-label agreement on the neuro umap embedding,
mahotas Haralick texture contrasts (PCam, neuro)."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
NP = Path("/home/sandbox/mega27-expansion/np")


def torchvision_neuro():
    import torch
    from torchvision import models
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    w = models.ResNet18_Weights.IMAGENET1K_V1
    rn = models.resnet18(weights=w)
    rn.fc = torch.nn.Identity()
    rn.eval()
    d = np.load(NP / "neuro_64.npz")
    def feats(X):
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                t = torch.from_numpy(X[i:i+128].astype(np.float32) / 255.0)
                t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear")
                mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                outs.append(rn((t - mean) / std).numpy())
        return np.concatenate(outs)
    Xtr, Xte = feats(d["Xtr"][:2000]), feats(d["Xte"][:1000])
    ytr, yte = d["ytr"][:2000], d["yte"][:1000]
    clf = LogisticRegression(max_iter=300)
    clf.fit(Xtr, ytr)
    bacc = balanced_accuracy_score(yte, clf.predict(Xte))
    dedup = json.load(open(OUT / "neuro" / "brain_mri_cnn_dedup_train.json"))["history"][-1]
    json.dump({"tool": "torchvision (pretrained ResNet18, ImageNet1K_V1)",
               "dataset": "neuro, 2000-train/1000-test features, logistic head",
               "transfer_bacc": round(float(bacc), 4),
               "scratch_dedup_cnn_reference": {"acc": dedup["acc"], "bacc": dedup["bacc"]},
               "note": "ImageNet transfer reference point for the from-scratch dedup CNN"},
              open(OUT / "neuro" / "neuro_torchvision_transfer.json", "w"), indent=1)


def hdbscan_neuro():
    import umap, hdbscan
    from sklearn.metrics import adjusted_rand_score
    d = np.load(NP / "neuro_64.npz")
    X, y = d["Xte"][:1000], d["yte"][:1000]
    Xf = X.reshape(len(X), -1).astype(np.float32) / 255.0
    emb = umap.UMAP(n_neighbors=15, min_dist=0.0, random_state=0, n_jobs=1).fit_transform(Xf)
    labels = hdbscan.HDBSCAN(min_cluster_size=25).fit_predict(emb)
    mask = labels >= 0
    ari = adjusted_rand_score(y[mask], labels[mask])
    json.dump({"tool": "hdbscan",
               "dataset": "neuro test umap embedding, 1000 images",
               "n_clustered": int(mask.sum()), "n_noise": int((labels < 0).sum()),
               "n_clusters": int(len(set(labels[mask]))),
               "ari_vs_true_labels": round(float(ari), 4),
               "note": "unsupervised cluster structure vs true classes (leakage caveat applies)"},
              open(OUT / "neuro" / "neuro_hdbscan_clusters.json", "w"), indent=1)


def _haralick(X, cap):
    import mahotas
    rows = []
    for i in range(min(cap, len(X))):
        g = np.moveaxis(X[i], 0, -1).mean(axis=2).astype(np.uint8)
        rows.append(mahotas.features.haralick(g).mean(axis=0))
    return np.array(rows)


def mahotas_pcam():
    from scipy.stats import mannwhitneyu
    d = np.load(NP / "pcam_test_4096.npz")
    X, y = d["X"][:800], d["y"][:800]
    H = _haralick(X, 800)
    contrast = H[:, 1]
    u = mannwhitneyu(contrast[y == 1], contrast[y == 0])
    json.dump({"tool": "mahotas (Haralick texture)",
               "dataset": "pcam test 800 patches",
               "mean_contrast_tumor": round(float(contrast[y == 1].mean()), 2),
               "mean_contrast_nontumor": round(float(contrast[y == 0].mean()), 2),
               "mwu_p": float(u.pvalue),
               "note": "Haralick texture contrast complements the wavelet/sharpness audits"},
              open(OUT / "cancer" / "pcam_mahotas_texture.json", "w"), indent=1)


def mahotas_neuro():
    from scipy.stats import kruskal
    d = np.load(NP / "neuro_64.npz")
    X, y = d["Xte"][:800], d["yte"][:800]
    H = _haralick(X, 800)
    contrast = H[:, 1]
    kw = kruskal(*[contrast[y == c] for c in range(4)])
    json.dump({"tool": "mahotas (Haralick texture)",
               "dataset": "neuro test 800 images",
               "kruskal_p": float(kw.pvalue),
               "note": "texture contrast across the four classes"},
              open(OUT / "neuro" / "neuro_mahotas_texture.json", "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    {"torchvision": torchvision_neuro, "hdbscan": hdbscan_neuro,
     "mahotas_pcam": mahotas_pcam, "mahotas_neuro": mahotas_neuro}[which]()
    print(which, "done", flush=True)
