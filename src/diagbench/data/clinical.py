"""Loaders for the five real clinical datasets (network-gated, cached)."""
from __future__ import annotations

import numpy as np

from .base import TabularDataset, fetch_text

WDBC_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
CLEVELAND_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
PARKINSONS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

WDBC_FEATURES = ["radius", "texture", "perimeter", "area", "smoothness",
                 "compactness", "concavity", "concave_points", "symmetry",
                 "fractal_dimension"]
WDBC_FEATURES = [f"{stat}_{f}" for stat in ("mean", "se", "worst")
                 for f in WDBC_FEATURES]

CLEVELAND_FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs",
                      "restecg", "thalach", "exang", "oldpeak", "slope",
                      "ca", "thal"]

PIMA_FEATURES = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                 "insulin", "bmi", "diabetes_pedigree", "age"]


def load_wdbc() -> TabularDataset:
    """Wisconsin Diagnostic Breast Cancer: 569 fine-needle aspirate cases."""
    text = fetch_text(WDBC_URL, "wdbc.data")
    rows = [ln.split(",") for ln in text.strip().splitlines() if ln.strip()]
    y = np.array([1 if r[1] == "M" else 0 for r in rows], dtype=np.int64)
    X = np.array([[float(v) for v in r[2:]] for r in rows], dtype=np.float32)
    return TabularDataset(
        name="wdbc", X=X, y=y, feature_names=WDBC_FEATURES,
        positive_label="malignant", source_url=WDBC_URL,
        citation="Street, Wolberg & Mangasarian (1993), UCI ML Repository",
    )


def load_cleveland() -> TabularDataset:
    """Cleveland Heart Disease: 303 patients, angiographic disease status."""
    text = fetch_text(CLEVELAND_URL, "cleveland.data")
    rows = [ln.split(",") for ln in text.strip().splitlines() if ln.strip()]
    X_raw, y = [], []
    for r in rows:
        X_raw.append([np.nan if v == "?" else float(v) for v in r[:13]])
        y.append(1 if float(r[13]) > 0 else 0)
    X = np.array(X_raw, dtype=np.float32)
    # median-impute the handful of missing ca/thal entries (train-safe: these
    # are column medians over the full public file, documented in the paper)
    for j in range(X.shape[1]):
        col = X[:, j]
        X[np.isnan(col), j] = np.nanmedian(col)
    return TabularDataset(
        name="cleveland", X=X, y=np.array(y, dtype=np.int64),
        feature_names=CLEVELAND_FEATURES, positive_label="heart_disease",
        source_url=CLEVELAND_URL,
        citation="Detrano et al. (1989), UCI ML Repository",
    )


def load_pima() -> TabularDataset:
    """Pima Indians Diabetes: 768 women >= 21y, OGTT-verified outcome."""
    text = fetch_text(PIMA_URL, "pima.csv")
    rows = [ln.split(",") for ln in text.strip().splitlines() if ln.strip()]
    X = np.array([[float(v) for v in r[:8]] for r in rows], dtype=np.float32)
    y = np.array([int(float(r[8])) for r in rows], dtype=np.int64)
    # zeros in glucose/bp/skin/insulin/bmi encode missing values -> NaN,
    # then median-impute (same documented convention as the loader)
    for j in (1, 2, 3, 4, 5):
        col = X[:, j]
        col[col == 0.0] = np.nan
        X[np.isnan(col), j] = np.nanmedian(col)
    return TabularDataset(
        name="pima", X=X, y=y, feature_names=PIMA_FEATURES,
        positive_label="diabetes", source_url=PIMA_URL,
        citation="Smith et al. (1988), NIDDK / UCI ML Repository",
    )


def load_parkinsons() -> TabularDataset:
    """Parkinson's voice recordings: 195 sustained-vowel phonations, 31 PwPD."""
    text = fetch_text(PARKINSONS_URL, "parkinsons.data")
    lines = text.strip().splitlines()
    header = lines[0].split(",")
    feat_idx = [i for i, h in enumerate(header)
                if h not in ("name", "status")]
    rows = [ln.split(",") for ln in lines[1:] if ln.strip()]
    X = np.array([[float(r[i]) for i in feat_idx] for r in rows],
                 dtype=np.float32)
    y = np.array([int(r[header.index("status")]) for r in rows],
                 dtype=np.int64)
    return TabularDataset(
        name="parkinsons", X=X, y=y,
        feature_names=[header[i] for i in feat_idx],
        positive_label="parkinsons", source_url=PARKINSONS_URL,
        citation="Little et al. (2007), UCI ML Repository",
    )


LOADERS = {
    "wdbc": load_wdbc,
    "cleveland": load_cleveland,
    "pima": load_pima,
    "parkinsons": load_parkinsons,
}


def load_dataset(name: str) -> TabularDataset:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(LOADERS)}")
    return LOADERS[name]()
