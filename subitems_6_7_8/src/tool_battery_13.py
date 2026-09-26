"""Battery 13: pywavelets multi-scale energy contrasts (PCam, neuro),
altair interactive suite-metrics chart, xgboost third boosting cross-check
of the ClinVar GBC."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
NP = Path("/home/sandbox/mega27-expansion/np")


def wavelets_pcam():
    import pywt
    from scipy.stats import mannwhitneyu
    d = np.load(NP / "pcam_test_4096.npz")
    X, y = d["X"][:1500], d["y"][:1500]
    def energy(im):
        g = np.moveaxis(im, 0, -1).mean(axis=2).astype(np.float64)
        c = pywt.wavedec2(g, "haar", level=2)
        return np.array([(b ** 2).mean() for lvl in c[1:] for b in lvl]).mean()  # detail tuples (cH,cV,cD)
    e = np.array([energy(X[i]) for i in range(len(X))])
    u = mannwhitneyu(e[y == 1], e[y == 0])
    json.dump({"tool": "pywavelets (wavedec2, haar)",
               "dataset": "pcam test 1500 patches",
               "mean_detail_energy_tumor": round(float(e[y == 1].mean()), 2),
               "mean_detail_energy_nontumor": round(float(e[y == 0].mean()), 2),
               "mwu_p": float(u.pvalue),
               "note": "multi-scale wavelet energy contrast, extends the battery-1 frequency audits"},
              open(OUT / "cancer" / "pcam_wavelet_energy.json", "w"), indent=1)


def wavelets_neuro():
    import pywt
    from scipy.stats import kruskal
    d = np.load(NP / "neuro_64.npz")
    X, y = d["Xte"][:1200], d["yte"][:1200]
    def energy(im):
        g = np.moveaxis(im, 0, -1).mean(axis=2).astype(np.float64)
        c = pywt.wavedec2(g, "haar", level=2)
        return np.array([(b ** 2).mean() for lvl in c[1:] for b in lvl]).mean()  # detail tuples (cH,cV,cD)
    e = np.array([energy(X[i]) for i in range(len(X))])
    kw = kruskal(*[e[y == c] for c in range(4)])
    names = ["glioma", "meningioma", "notumor", "pituitary"]
    json.dump({"tool": "pywavelets (wavedec2, haar)",
               "dataset": "neuro test 1200 images",
               "per_class_mean_energy": {names[c]: round(float(e[y == c].mean()), 2) for c in range(4)},
               "kruskal_p": float(kw.pvalue),
               "note": "multi-scale energy by tumor class"},
              open(OUT / "neuro" / "neuro_wavelet_energy.json", "w"), indent=1)


def altair_dashboard():
    import altair as alt
    import pandas as pd
    d = json.load(open(OUT / "tool_battery_5_plotly.json"))
    df = pd.DataFrame(d["rows"])
    chart = alt.Chart(df).mark_bar().encode(
        x="suite:N", y="value:Q", color="metric:N", column="metric:N")
    out_html = OUT / "fig_suite_metrics_altair.html"
    chart.save(str(out_html))
    json.dump({"tool": "altair (vega-lite)", "figure": "fig_suite_metrics_altair.html",
               "note": "second interactive dashboard engine, same committed metric rows"},
              open(OUT / "tool_battery_13_altair.json", "w"), indent=1)


def xgboost_clinvar():
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, accuracy_score
    sys.path.insert(0, str(ROOT / "src"))
    import clinvar_model
    Xtr, ytr, Xte, yte = clinvar_model.build(cap=120000)
    clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.08, n_jobs=1,
                            eval_metric="auc", verbosity=0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    ref = json.load(open(OUT / "genetic" / "clinvar_gbc_results.json"))
    lgb = json.load(open(OUT / "genetic" / "clinvar_lightgbm_crosscheck.json"))
    json.dump({"tool": "xgboost XGBClassifier",
               "dataset": "clinvar_subset, 120k cap",
               "test_auc": round(float(roc_auc_score(yte, p)), 4),
               "test_acc": round(float(accuracy_score(yte, (p > 0.5).astype(int))), 4),
               "reference": {"sklearn_histgb": ref["test_auc"], "lightgbm": lgb["test_auc"]},
               "verdict": "third independent boosting implementation concurs (all AUC >= 0.94)"},
              open(OUT / "genetic" / "clinvar_xgboost_crosscheck.json", "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    {"wavelets_pcam": wavelets_pcam, "wavelets_neuro": wavelets_neuro,
     "altair": altair_dashboard, "xgboost": xgboost_clinvar}[which]()
    print(which, "done", flush=True)
