"""Battery 11: pandera schema validation of the ClinVar stream, pingouin
effect sizes for the property contrasts, scikit-posthocs Dunn posthoc,
cleanlab multiclass census on the neuro dedup probs, pycm BreakHis stats."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
NP = Path("/home/sandbox/mega27-expansion/np")


def pandera_clinvar():
    import pandas as pd
    import pandera.pandas as pa
    sys.path.insert(0, str(ROOT / "src"))
    import loaders
    rows, seen = [], set()
    for r in loaders.iter_clinvar_subset():
        if r["variation_id"] in seen:
            continue
        seen.add(r["variation_id"])
        rows.append({"variation_id": int(r["variation_id"]), "label": r["label"],
                     "chrom": r["chrom"], "pos": int(r["pos"]) if r["pos"].isdigit() else None,
                     "ref_len": len(r["ref"]), "alt_len": len(r["alt"])})
        if len(rows) >= 100000:
            break
    df = pd.DataFrame(rows)
    schema = pa.DataFrameSchema({
        "variation_id": pa.Column(int, pa.Check.gt(0)),
        "label": pa.Column(int, pa.Check.isin([0, 1])),
        "chrom": pa.Column(str),
        "pos": pa.Column(pd.Int64Dtype(), pa.Check.gt(0), nullable=True),
        "ref_len": pa.Column(int, pa.Check.gt(0)),
        "alt_len": pa.Column(int, pa.Check.gt(0)),
    })
    schema.validate(df)
    json.dump({"tool": "pandera DataFrameSchema",
               "dataset": "clinvar_subset, 100k streamed rows",
               "n_validated": int(len(df)),
               "checks": ["variation_id>0", "label in {0,1}", "pos>0 nullable", "ref_len>0", "alt_len>0"],
               "verdict": "schema valid",
               "note": "data-contract gate on the modeling stream"},
              open(OUT / "genetic" / "clinvar_pandera_schema.json", "w"), indent=1)


def _props(tag):
    d = json.load(open(OUT / tag / f"tool_battery_1_{'breakhis' if tag=='cancer' else 'breakhis'}_props.json")) if tag == "cancer" else None


def pingouin_effects():
    import pingouin as pg
    from skimage import filters
    d = np.load(NP / "pcam_test_4096.npz")
    X, y = d["X"][:2000], d["y"][:2000]
    sharp = [filters.laplace(X[i].astype(np.float32).mean(axis=0)).var() for i in range(len(X))]
    sharp = np.array(sharp)
    cohen = pg.compute_effsize(sharp[y == 1], sharp[y == 0], eftype="cohen")
    hedges = pg.compute_effsize(sharp[y == 1], sharp[y == 0], eftype="hedges")
    mwu = pg.mwu(sharp[y == 1], sharp[y == 0])
    pcol = [c for c in mwu.columns if "p" in c.lower()][0]
    json.dump({"tool": "pingouin",
               "dataset": "pcam test, 2000 patches, Laplace-variance sharpness",
               "cohens_d": round(float(cohen), 4), "hedges_g": round(float(hedges), 4),
               "mwu_p": float(mwu[pcol].iloc[0]),
               "note": "effect sizes complement the battery-1 Mann-Whitney contrast"},
              open(OUT / "cancer" / "pcam_pingouin_effects.json", "w"), indent=1)


def posthoc_props():
    import scikit_posthocs as sp
    from skimage import filters
    d = np.load(NP / "neuro_64.npz")
    X, y = d["Xte"], d["yte"]
    sharp = np.array([filters.laplace(X[i].astype(np.float32).mean(axis=0)).var() for i in range(len(X))])
    groups = [sharp[y == c] for c in range(4)]
    dunn = sp.posthoc_dunn(groups, p_adjust="holm")
    names = ["glioma", "meningioma", "notumor", "pituitary"]
    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            pairs.append({"a": names[i], "b": names[j], "p_holm": float(dunn.iloc[i, j])})
    json.dump({"tool": "scikit-posthocs (Dunn, Holm-adjusted)",
               "dataset": "neuro test sharpness by class",
               "pairs": pairs,
               "n_significant_05": int(sum(1 for p in pairs if p["p_holm"] < 0.05)),
               "note": "posthoc follow-up to the battery-1 Kruskal-Wallis class contrast"},
              open(OUT / "neuro" / "neuro_posthoc_dunn.json", "w"), indent=1)


def cleanlab_neuro():
    from cleanlab.filter import find_label_issues
    d = np.load(OUT / "neuro" / "brain_mri_cnn_dedup_probs.npz")
    y, P = d["y_true"], d["probs"]
    issues = find_label_issues(y, P, return_indices_ranked_by="self_confidence")
    json.dump({"tool": "cleanlab (multiclass)",
               "dataset": "neuro dedup held-out probs",
               "n_test": int(len(y)), "n_flagged": int(len(issues)),
               "noise_rate_pct": round(100 * len(issues) / len(y), 2),
               "top20_flags": [{"index": int(i), "given": int(y[i])} for i in issues[:20]],
               "note": "model-conditional label-issue candidates on the leakage-controlled model"},
              open(OUT / "neuro" / "neuro_dedup_label_census.json", "w"), indent=1)


def pycm_breakhis():
    from pycm import ConfusionMatrix
    d = np.load(OUT / "cancer" / "breakhis_cnn_probs.npz")
    y, p = d["y_true"], d["prob_pos"]
    pred = (p > 0.5).astype(int)
    cm = ConfusionMatrix(y.tolist(), pred.tolist())
    json.dump({"tool": "pycm ConfusionMatrix",
               "dataset": "breakhis held-out probs (patient-level split)",
               "matrix": {"TP": int(cm.TP[1]), "FP": int(cm.FP[1]), "TN": int(cm.TN[1]), "FN": int(cm.FN[1])},
               "acc": round(float(cm.overall_stat["Overall ACC"]), 4),
               "kappa": round(float(cm.overall_stat["Kappa"]), 4),
               "f1_macro": round(float(cm.overall_stat["F1 Macro"]), 4),
               "note": "confusion stats for the split-verified BreakHis model"},
              open(OUT / "cancer" / "breakhis_pycm_stats.json", "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    {"pandera": pandera_clinvar, "pingouin": pingouin_effects,
     "posthoc": posthoc_props, "cleanlab_neuro": cleanlab_neuro,
     "pycm_breakhis": pycm_breakhis}[which]()
    print(which, "done", flush=True)
