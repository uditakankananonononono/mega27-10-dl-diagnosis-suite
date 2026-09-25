"""Do census-flagged images differ measurably from unflagged ones?
Computes per-image sharpness (variance of Laplacian), RMS contrast and
entropy with scikit-image for flagged vs matched unflagged samples, and
tests the difference (scipy Mann-Whitney U). A measurable difference
supports the census being a property of the images, not of the estimator."""
import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from skimage.filters import laplace
from skimage.measure import shannon_entropy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.data import NpyDataset

ROOT = Path(__file__).resolve().parent.parent


def props(arr):
    g = arr.mean(0) if arr.ndim == 3 else arr   # gray
    g = g / 255.0
    return {"sharpness": float(laplace(g).var()),
            "contrast": float(g.std()),
            "entropy": float(shannon_entropy((g * 255).astype(np.uint8)))}


def main(disease: str, prefix: str, n_control: int = 500):
    ds = NpyDataset(prefix, train=False)
    census = json.load(open(ROOT / "results" / disease / "label_noise_census.json"))
    split = json.load(open(ROOT / "results" / disease / "split.json"))
    tr = np.array(split["train"])
    flagged_ids = set(census["flagged_ids"])
    ids = np.array([str(i) for i in ds.ids])
    flag_mask = np.array([i in flagged_ids for i in ids[tr]])
    flag_idx, clean_idx = tr[flag_mask], tr[~flag_mask]
    rng = np.random.default_rng(0)
    clean_sub = rng.choice(clean_idx, min(n_control, len(clean_idx)), replace=False)

    out = {"disease": disease, "n_flagged": len(flag_idx), "n_control": len(clean_sub)}
    for name, idxs in (("flagged", flag_idx), ("control", clean_sub)):
        P = [props(np.asarray(ds.x[i], dtype=np.float32)) for i in idxs]
        out[name] = {k: {"mean": round(float(np.mean([p[k] for p in P])), 6),
                         "median": round(float(np.median([p[k] for p in P])), 6)}
                     for k in P[0]}
        out.setdefault("_raw", {})[name] = {k: [p[k] for p in P] for k in P[0]}
    tests = {}
    for k in ("sharpness", "contrast", "entropy"):
        u = mannwhitneyu(out["_raw"]["flagged"][k], out["_raw"]["control"][k],
                         alternative="two-sided")
        tests[k] = {"U": float(u.statistic), "p": float(u.pvalue)}
    out["mann_whitney"] = tests
    del out["_raw"]
    dest = ROOT / "results" / disease / "flagged_image_properties.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(disease, json.dumps(tests, indent=1))


if __name__ == "__main__":
    main("malaria", str(ROOT / "data" / "malaria" / "malaria48"))
    main("pneumonia", str(ROOT / "data" / "pneumonia" / "cxr_train"))
