"""shap feature attribution on wdbc's best model + imbalanced-learn SMOTE
ablation on the 3 most imbalanced binary panel datasets.
Output analyses/shap_smote.json"""
import json, os, sys, warnings
sys.path.insert(0, "src")
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

out = {}
# ---- shap on wdbc ----
import shap
from diagbench.data.clinical import load_wdbc
ds = load_wdbc()
clf = make_pipeline(StandardScaler(),
                    LogisticRegression(max_iter=2000, class_weight="balanced"))
clf.fit(ds.X, ds.y)
lr = clf.named_steps["logisticregression"]
sc = clf.named_steps["standardscaler"]
Xs = sc.transform(ds.X)
expl = shap.LinearExplainer(lr, Xs)
sv = expl.shap_values(Xs[:200])
names = ds.feature_names if getattr(ds, "feature_names", None) else \
        [f"f{i}" for i in range(ds.X.shape[1])]
mean_abs = np.abs(sv).mean(axis=0)
top = np.argsort(mean_abs)[::-1][:10]
out["shap_wdbc"] = {"tool": "shap",
    "method": "LinearExplainer on StandardScaler-transformed wdbc, mean |SHAP| over first 200 patients",
    "top10_features": [{"feature": str(names[i]),
                        "mean_abs_shap": float(mean_abs[i])} for i in top]}

# ---- imblearn SMOTE ablation ----
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from diagbench.data.panel import PANEL_LOADERS
imb = []
for name, fn in PANEL_LOADERS.items():
    d = fn()
    pos = int((d.y == 1).sum()); neg = int((d.y == 0).sum())
    if min(pos, neg) >= 6:
        imb.append((min(pos, neg) / max(pos, neg), name, d))
imb.sort()
res = []
for ratio, name, d in imb[:3]:
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    base = make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"))
    sm = ImbPipeline([("sc", StandardScaler()), ("smote", SMOTE(random_state=0)),
                      ("lr", LogisticRegression(max_iter=2000))])
    p0 = cross_val_predict(base, d.X, d.y, cv=cv, method="predict_proba")[:, 1]
    p1 = cross_val_predict(sm, d.X, d.y, cv=cv, method="predict_proba")[:, 1]
    res.append({"dataset": name, "minority_fraction": float(ratio),
                "auc_class_weight": float(roc_auc_score(d.y, p0)),
                "auc_smote": float(roc_auc_score(d.y, p1))})
    print(f"DONE smote {name}", flush=True)
out["imblearn_smote"] = {"tool": "imbalanced-learn",
    "method": "5-fold CV AUC, class-weighted logreg vs SMOTE+logreg, 3 most imbalanced binary panel datasets",
    "results": res}
json.dump(out, open("analyses/shap_smote.json", "w"), indent=1)
print("SHAP_SMOTE_DONE", flush=True)
