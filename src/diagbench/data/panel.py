"""Extended clinical diagnosis panel: verified UCI datasets, generic loaders.

Every URL here was HEAD-verified 200 on 2026-09-24. Only diagnosis-relevant
datasets are included (no spam/sonar padding). Each loader returns a
TabularDataset with numeric features (categoricals factorized, missing
median-imputed) and a binarized diagnosis label.
"""
from __future__ import annotations

import io
import numpy as np

from .base import TabularDataset, fetch_text

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"


def _to_matrix(rows, label_col, label_fn, drop=(), factorize_maps=None):
    cols, y = [], []
    n_cols = len(rows[0])
    keep = [j for j in range(n_cols) if j != label_col and j not in drop]
    feats = [[] for _ in keep]
    for r in rows:
        if len(r) != n_cols:
            continue
        try:
            y.append(label_fn(r[label_col]))
        except (ValueError, IndexError):
            continue
        for k, j in enumerate(keep):
            feats[k].append(r[j])
    X_cols = []
    names = []
    for k, col in enumerate(feats):
        vals = []
        numeric = True
        for v in col:
            try:
                vals.append(float(v) if v not in ("?", "", "NA") else np.nan)
            except ValueError:
                numeric = False
                break
        if numeric:
            X_cols.append(np.array(vals, dtype=np.float32))
        else:
            uniq = {v: i for i, v in enumerate(sorted(set(col)))}
            X_cols.append(np.array([uniq[v] for v in col], dtype=np.float32))
        names.append(f"x{k}")
    X = np.stack(X_cols, axis=1)
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            X[np.isnan(col), j] = np.nanmedian(col)
    return X.astype(np.float32), np.array(y, dtype=np.int64), names


def _load(name, url, sep, label_col, label_fn, skip=0, drop=(),
          positive="positive", citation="UCI ML Repository"):
    text = fetch_text(url, f"{name}.raw")
    lines = [ln for ln in text.strip().splitlines()[skip:] if ln.strip()]
    rows = [ln.split(sep) for ln in lines]
    rows = [[c.strip() for c in r] for r in rows]
    X, y, names = _to_matrix(rows, label_col, label_fn, drop=drop)
    return TabularDataset(name=name, X=X, y=y, feature_names=names,
                          positive_label=positive, source_url=url,
                          citation=citation)


def load_wpbc():
    return _load("wpbc", f"{UCI}/breast-cancer-wisconsin/wpbc.data", ",",
                 1, lambda v: 1 if v == "R" else 0,
                 positive="recurrence",
                 citation="Wolberg et al., UCI (prognostic breast cancer)")


def load_wisconsin_original():
    return _load("wisconsin_original",
                 f"{UCI}/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                 ",", 10, lambda v: 1 if v == "4" else 0, drop=(0,),
                 positive="malignant",
                 citation="Wolberg & Mangasarian (1990), UCI")


def load_heart_site(site):
    ds = _load(f"heart_{site}", f"{UCI}/heart-disease/processed.{site}.data",
               ",", 13, lambda v: 1 if float(v) > 0 else 0,
               positive="heart_disease",
               citation=f"Detrano et al. (1989), UCI ({site} site)")
    return ds


def load_hepatitis():
    return _load("hepatitis", f"{UCI}/hepatitis/hepatitis.data", ",",
                 0, lambda v: 1 if v == "1" else 0,
                 positive="die", citation="UCI ML Repository")


def load_statlog_heart():
    return _load("statlog_heart", f"{UCI}/statlog/heart/heart.dat", None,
                 13, lambda v: 1 if v == "2" else 0,
                 positive="heart_disease", citation="StatLog / UCI")


def load_spectf():
    """SPECTF: official train+test concatenated (documented split preserved
    via the 'fold' attribute is out of scope; we resplit seeded)."""
    tr = fetch_text(f"{UCI}/spect/SPECTF.train", "spectf_train.raw")
    te = fetch_text(f"{UCI}/spect/SPECTF.test", "spectf_test.raw")
    lines = [ln for ln in (tr.strip() + "\n" + te.strip()).splitlines()
             if ln.strip()]
    rows = [[c.strip() for c in ln.split(",")] for ln in lines]
    X, y, names = _to_matrix(rows, 0, lambda v: int(float(v)))
    return TabularDataset(name="spectf", X=X, y=y, feature_names=names,
                          positive_label="abnormal",
                          source_url=f"{UCI}/spect/",
                          citation="Kurgan et al. (2001), UCI")


def load_haberman():
    return _load("haberman", f"{UCI}/haberman/haberman.data", ",",
                 3, lambda v: 1 if v == "2" else 0,
                 positive="died_5y", citation="Haberman (1976), UCI")


def load_lymphography():
    ds = _load("lymphography", f"{UCI}/lymphography/lymphography.data", ",",
               0, lambda v: 1 if v in ("3", "4") else 0,
               positive="malignant_lymphoma", citation="UCI ML Repository")
    return ds


def load_dermatology():
    ds = _load("dermatology", f"{UCI}/dermatology/dermatology.data", ",",
               34, lambda v: 0 if v == "1" else 1,
               positive="non_psoriasis_erythemato_squamous",
               citation="Ilter & Guvenir (1998), UCI")
    return ds


def load_primary_tumor():
    return _load("primary_tumor", f"{UCI}/primary-tumor/primary-tumor.data",
                 ",", 0, lambda v: 1, positive="na",
                 citation="UCI ML Repository")  # multiclass-only; skipped in panel


def load_mammographic():
    return _load("mammographic",
                 f"{UCI}/mammographic-masses/mammographic_masses.data", ",",
                 5, lambda v: 1 if v == "1" else 0,
                 positive="malignant", citation="Elter et al. (2007), UCI")


def load_echocardiogram():
    ds = _load("echocardiogram",
               f"{UCI}/echocardiogram/echocardiogram.data", ",",
               1, lambda v: 1 if v == "1" else 0, drop=(0, 10, 11),
               positive="alive_1y", citation="UCI ML Repository")
    return ds


def load_ann_thyroid():
    tr = fetch_text(f"{UCI}/thyroid-disease/ann-train.data", "ann_train.raw")
    te = fetch_text(f"{UCI}/thyroid-disease/ann-test.data", "ann_test.raw")
    lines = [ln for ln in (tr.strip() + "\n" + te.strip()).splitlines()
             if ln.strip()]
    rows = [ln.split() for ln in lines]
    rows = [[c for c in r] for r in rows]
    X, y, names = _to_matrix(rows, 21, lambda v: 0 if v == "3" else 1)
    return TabularDataset(name="ann_thyroid", X=X, y=y, feature_names=names,
                          positive_label="thyroid_dysfunction",
                          source_url=f"{UCI}/thyroid-disease/",
                          citation="Quinlan, UCI (ANN thyroid)")


def load_new_thyroid():
    return _load("new_thyroid",
                 f"{UCI}/thyroid-disease/new-thyroid.data", ",",
                 0, lambda v: 0 if v == "1" else 1,
                 positive="thyroid_dysfunction", citation="UCI ML Repository")


def load_arrhythmia():
    ds = _load("arrhythmia", f"{UCI}/arrhythmia/arrhythmia.data", ",",
               279, lambda v: 1 if v != "1" else 0, drop=(),
               positive="arrhythmia",
               citation="Guvenir et al. (1997), UCI")
    return ds


def load_cervical():
    ds = _load("cervical",
               f"{UCI}/00383/risk_factors_cervical_cancer.csv", ",",
               35, lambda v: 1 if v == "1" else 0, skip=1,
               positive="biopsy_positive",
               citation="Fernandes et al. (2017), UCI")
    return ds


def load_messidor():
    text = fetch_text(f"{UCI}/00329/messidor_features.arff", "messidor.raw")
    lines = [ln for ln in text.splitlines()
             if ln.strip() and not ln.startswith("@")]
    rows = [[c.strip() for c in ln.split(",")] for ln in lines]
    X, y, names = _to_matrix(rows, 19, lambda v: int(float(v)))
    return TabularDataset(name="messidor", X=X, y=y, feature_names=names,
                          positive_label="diabetic_retinopathy",
                          source_url=f"{UCI}/00329/messidor_features.arff",
                          citation="Antal & Hajdu (2014), UCI")


def load_fertility():
    ds = _load("fertility", f"{UCI}/00244/fertility_Diagnosis.txt", ",",
               9, lambda v: 1 if v.strip().strip('"') == "O" else 0,
               positive="altered", citation="Gil et al. (2012), UCI")
    return ds


def load_hcv():
    ds = _load("hcv", f"{UCI}/00571/hcvdat0.csv", ",",
               1, lambda v: 0 if v.strip().strip('"').startswith("0=") else 1,
               skip=1, drop=(0,), positive="hepatitis_or_fibrosis",
               citation="Lichtinghagen et al. / Hoffmann et al. (2018), UCI")
    return ds


def load_ilpd():
    ds = _load("ilpd",
               f"{UCI}/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
               ",", 10, lambda v: 1 if v == "1" else 0,
               positive="liver_disease",
               citation="Ramana & Venkateswarlu, UCI")
    return ds


def load_diabetes_upload():
    ds = _load("diabetes_early_risk",
               f"{UCI}/00529/diabetes_data_upload.csv", ",",
               16, lambda v: 1 if v.strip() == "Positive" else 0, skip=1,
               positive="diabetes_risk", citation="Islam et al. (2020), UCI")
    return ds


PANEL_LOADERS = {
    "wpbc": load_wpbc,
    "wisconsin_original": load_wisconsin_original,
    "heart_hungarian": lambda: load_heart_site("hungarian"),
    "heart_switzerland": lambda: load_heart_site("switzerland"),
    "heart_va": lambda: load_heart_site("va"),
    "hepatitis": load_hepatitis,
    "statlog_heart": load_statlog_heart,
    "spectf": load_spectf,
    "haberman": load_haberman,
    "lymphography": load_lymphography,
    "dermatology": load_dermatology,
    "mammographic": load_mammographic,
    "echocardiogram": load_echocardiogram,
    "ann_thyroid": load_ann_thyroid,
    "new_thyroid": load_new_thyroid,
    "arrhythmia": load_arrhythmia,
    "cervical": load_cervical,
    "messidor": load_messidor,
    "fertility": load_fertility,
    "hcv": load_hcv,
    "ilpd": load_ilpd,
    "diabetes_early_risk": load_diabetes_upload,
}
