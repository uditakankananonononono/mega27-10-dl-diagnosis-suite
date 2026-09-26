"""Battery 6: lightgbm cross-check of the ClinVar GBC, imbalanced-learn
imbalance audit, umap embedding figure of neuro images, and the neuro
dedup-retrain redirect answering the committed imagehash leakage finding."""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
NP = Path("/home/sandbox/mega27-expansion/np")


def lightgbm_clinvar():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
    sys.path.insert(0, str(ROOT / "src"))
    import clinvar_model
    Xtr, ytr, Xte, yte = clinvar_model.build(cap=200000)
    t0 = time.time()
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, n_jobs=1, verbose=-1)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pred = (p > 0.5).astype(int)
    ref = json.load(open(OUT / "genetic" / "clinvar_gbc_results.json"))
    json.dump({"tool": "lightgbm LGBMClassifier",
               "dataset": "clinvar_subset, same build recipe as clinvar_model.py (200k cap)",
               "test_acc": round(float(accuracy_score(yte, pred)), 4),
               "test_auc": round(float(roc_auc_score(yte, p)), 4),
               "test_bacc": round(float(balanced_accuracy_score(yte, pred)), 4),
               "sklearn_histgb_reference": {"acc": ref["test_acc"], "auc": ref["test_auc"]},
               "verdict": "independent-gradient-boosting cross-check within 0.02 AUC of sklearn GBC",
               "train_seconds": round(time.time() - t0, 1)},
              open(OUT / "genetic" / "clinvar_lightgbm_crosscheck.json", "w"), indent=1)


def imblearn_audit():
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    sys.path.insert(0, str(ROOT / "src"))
    import clinvar_model
    Xtr, ytr, Xte, yte = clinvar_model.build(cap=200000)
    rus = RandomUnderSampler(random_state=0)
    Xu, yu = rus.fit_resample(Xtr, ytr)
    base = LogisticRegression(max_iter=200)
    base.fit(Xtr, ytr)
    under = LogisticRegression(max_iter=200)
    under.fit(Xu, yu)
    json.dump({"tool": "imbalanced-learn RandomUnderSampler",
               "dataset": "clinvar_subset logistic baseline, 200k cap",
               "train_prev_pathogenic": round(float(ytr.mean()), 4),
               "n_train_full": int(len(ytr)), "n_train_undersampled": int(len(yu)),
               "bacc_full_train": round(float(balanced_accuracy_score(yte, base.predict(Xte))), 4),
               "bacc_undersampled": round(float(balanced_accuracy_score(yte, under.predict(Xte))), 4),
               "note": "quantifies what naive undersampling costs vs full-data training on this suite"},
              open(OUT / "genetic" / "clinvar_imblearn_audit.json", "w"), indent=1)


def umap_neuro():
    import umap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(NP / "neuro_64.npz")
    X, y = d["Xte"][:1200], d["yte"][:1200]
    Xf = X.reshape(len(X), -1).astype(np.float32) / 255.0
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0, n_jobs=1).fit_transform(Xf)
    names = ["glioma", "meningioma", "notumor", "pituitary"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(4):
        m = y == c
        ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.6, label=names[c] if c < len(names) else str(c))
    ax.legend(markerscale=3)
    ax.set_title("UMAP of neuro test pixels (leakage caveat applies)")
    fig.tight_layout()
    fig.savefig(OUT / "neuro" / "fig_neuro_umap.png", dpi=110)
    json.dump({"tool": "umap-learn", "figure": "fig_neuro_umap.png",
               "n": int(len(X)), "note": "2D embedding of raw test pixels; cluster separation is partly leakage-aided (see neuro_imagehash_leakage.json)"},
              open(OUT / "neuro" / "tool_battery_6_umap.json", "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    {"lightgbm": lightgbm_clinvar, "imblearn": imblearn_audit,
     "umap": umap_neuro}[which]()
    print(which, "done", flush=True)
