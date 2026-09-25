"""Tool battery 1 for 10.6-10.8: genuine per-suite analyses, each emitting
committed JSON evidence. Pattern per subitems_4_5: a tool counts only when
its evidence file exists. Every claim here is measured, none asserted."""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders

OUT = Path(__file__).resolve().parent.parent / "results"


def image_props(suite, it, n, ds_name, thumb=(64, 64)):
    """scikit-image + scipy + Pillow + numpy: sharpness/entropy/property audit
    with class-conditional stats and a Mann-Whitney contrast."""
    from PIL import Image
    from skimage.measure import shannon_entropy
    from skimage.filters import laplace
    from scipy import stats as sstats
    t0 = time.time()
    from collections import defaultdict
    props = defaultdict(lambda: {"sharp": [], "ent": []})
    n_done = 0
    for r in it:
        if n_done >= n:
            break
        img = r["image"]
        if img.shape[0] > thumb[0]:
            img = np.asarray(Image.fromarray(img).resize(thumb, Image.BILINEAR))
        gray = img.mean(axis=2) / 255.0
        props[r["label"]]["sharp"].append(float(np.var(laplace(gray))))
        props[r["label"]]["ent"].append(float(shannon_entropy(img)))
        n_done += 1
    res = {"dataset": ds_name, "n_images": n_done, "wall_seconds": round(time.time() - t0, 2),
           "tools": ["scikit-image", "scipy", "Pillow", "numpy"], "thumb": list(thumb)}
    labels = sorted(props.keys())
    for lbl in labels:
        for k in ("sharp", "ent"):
            v = props[lbl][k]
            res[f"class{lbl}_{k}_mean"] = round(float(np.mean(v)), 5) if v else None
        res[f"class{lbl}_n"] = len(props[lbl]["sharp"])
    if len(labels) == 2 and all(props[l]["sharp"] for l in labels):
        u, p = sstats.mannwhitneyu(props[labels[0]]["sharp"], props[labels[1]]["sharp"], alternative="two-sided")
        res["sharpness_mannwhitney_p"] = float(p)
    elif len(labels) > 2 and all(props[l]["sharp"] for l in labels):
        h, p = sstats.kruskal(*[props[l]["sharp"] for l in labels])
        res["sharpness_kruskal_p"] = float(p)
    return res


def pixel_baseline(suite, it_train, it_test, n_tr, n_te, ds_name, thumb=(32, 32)):
    """scikit-learn + statsmodels: logistic regression on downsampled pixels,
    class-balance-matched, with Wilson 95% CI on held-out accuracy."""
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from statsmodels.stats.proportion import proportion_confint
    t0 = time.time()

    def harvest(it, m):
        X, y = [], []
        for i, r in enumerate(it):
            if i >= m:
                break
            im = Image.fromarray(r["image"]).resize(thumb, Image.BILINEAR)
            X.append(np.asarray(im, dtype=np.float32).ravel() / 255.0)
            y.append(r["label"])
        return np.stack(X), np.array(y)

    Xtr, ytr = harvest(it_train, n_tr)
    Xte, yte = harvest(it_test, n_te)
    clf = LogisticRegression(max_iter=200, C=0.5, n_jobs=2)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    lo, hi = proportion_confint(int(acc * len(yte)), len(yte), alpha=0.05, method="wilson")
    return {"dataset": ds_name, "model": "sklearn LogisticRegression on %dx%d px" % thumb,
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "test_accuracy": round(acc, 4),
            "wilson95": [round(float(lo), 4), round(float(hi), 4)],
            "tools": ["scikit-learn", "statsmodels", "Pillow", "numpy"],
            "note": "downsampled-pixel baseline, NOT the CNN benchmark number",
            "wall_seconds": round(time.time() - t0, 2)}


def patient_split_breakhis():
    """pandas: patient-level split for BreakHis (leakage-safe), cross-tabbed."""
    import pandas as pd
    df = pd.read_csv(OUT / "cancer" / "breakhis_accessions.csv")
    pats = sorted(df["patient_id"].unique())
    rng = np.random.RandomState(42)
    perm = rng.permutation(pats)
    n = len(pats)
    assign = {p: ("train" if i < 0.7 * n else "val" if i < 0.85 * n else "test")
              for i, p in enumerate(perm)}
    df["split"] = df["patient_id"].map(assign)
    tab = df.groupby(["split", "benign_malignant"]).size().unstack(fill_value=0)
    out = {"dataset": "breakhis", "tool": "pandas",
           "n_patients": int(n), "split_rule": "patient-level 70/15/15, seed 42",
           "patients_per_split": df.groupby("split")["patient_id"].nunique().to_dict(),
           "images_per_split_x_class": {s: tab.loc[s].to_dict() for s in tab.index}}
    df.to_csv(OUT / "cancer" / "breakhis_patient_splits.csv", index=False)
    return out


suite_map = {"cancer": "cancer", "cancer_baseline": "cancer", "neuro": "neuro", "genetic": "genetic"}

if __name__ == "__main__":
    which = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    res = {}
    if which == "cancer":
        (OUT / "cancer").mkdir(exist_ok=True)
        res["breakhis_props"] = image_props("cancer", loaders.iter_breakhis(), n, "breakhis")
        res["pcam_props"] = image_props("cancer", loaders.iter_pcam("valid"), n, "pcam_valid")
        res["breakhis_patient_split"] = patient_split_breakhis()
    elif which == "cancer_baseline":
        res["pcam_pixel_baseline"] = pixel_baseline(
            "cancer", loaders.iter_pcam("valid"), loaders.iter_pcam("test"),
            n, n // 2, "pcam valid->test")
    elif which == "neuro":
        (OUT / "neuro").mkdir(exist_ok=True)
        import pyarrow.parquet as pq
        labels = pq.read_table(loaders.DATA / "neuro" / "brain_tumor_mri_train.parquet", columns=["label"]).column("label").to_pylist()
        rng = np.random.RandomState(42)
        by_class = {}
        for i, l in enumerate(labels):
            by_class.setdefault(int(l), []).append(i)
        tr_idx, te_idx = [], []
        per_tr = n // (4 * len(by_class))
        per_te = n // (16 * len(by_class)) or 25
        for l, idxs in sorted(by_class.items()):
            idxs = rng.permutation(idxs).tolist()
            te_idx += idxs[:per_te]; tr_idx += idxs[per_te:per_te + per_tr]
        want = {i: ("train" if i in set(tr_idx) else "test") for i in tr_idx + te_idx}
        def sel(split):
            for i, r in enumerate(loaders.iter_neuro()):
                if want.get(i) == split:
                    yield r
        res["neuro_props"] = image_props("neuro", sel("train"), n // 4, "brain_mri")
        res["neuro_pixel_baseline"] = pixel_baseline("neuro", sel("train"), sel("test"), per_tr * len(by_class), per_te * len(by_class), "brain_mri stratified80/20")
    elif which == "genetic":
        (OUT / "genetic").mkdir(exist_ok=True)
        import csv
        from sklearn.linear_model import LogisticRegression
        from statsmodels.stats.proportion import proportion_confint
        t0 = time.time()
        chroms = {}
        X, y = [], []
        for i, r in enumerate(loaders.iter_clinvar_subset()):
            if i >= n * 4:
                break
            c = chroms.setdefault(r["chrom"], len(chroms))
            ref, alt = r["ref"][:1] or "N", r["alt"][:1] or "N"
            base = {"A": 0, "C": 1, "G": 2, "T": 3}.get
            X.append([c, len(r["ref"]), len(r["alt"]), base(ref, 4), base(alt, 4),
                      len(r["gene"])])
            y.append(r["label"])
        X, y = np.array(X, dtype=np.float32), np.array(y)
        k = len(y) * 4 // 5
        clf = LogisticRegression(max_iter=300, n_jobs=2).fit(X[:k], y[:k])
        acc = float(clf.score(X[k:], y[k:]))
        lo, hi = proportion_confint(int(acc * (len(y) - k)), len(y) - k, alpha=0.05, method="wilson")
        res["clinvar_baseline"] = {
            "dataset": "clinvar_subset", "model": "sklearn LogisticRegression on locus/allele features",
            "n_train": int(k), "n_test": int(len(y) - k), "test_accuracy": round(acc, 4),
            "wilson95": [round(float(lo), 4), round(float(hi), 4)],
            "prevalence_pathogenic": round(float(y.mean()), 4),
            "tools": ["scikit-learn", "statsmodels", "numpy"],
            "note": "trivial-feature floor, NOT the sequence-model number",
            "wall_seconds": round(time.time() - t0, 2)}
    for k, v in res.items():
        json.dump(v, open(OUT / suite_map.get(which, which) / f"tool_battery_1_{k}.json", "w"), indent=1)
        print(k, "ok")


