"""xgboost baseline across the 22-dataset UCI panel: same 5-fold stratified
CV protocol as the deep models. Output analyses/xgboost_baseline.json."""
import json, os, sys, time, warnings
sys.path.insert(0, "src")
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from diagbench.data.panel import PANEL_LOADERS

out = {}
if os.path.exists("analyses/xgboost_baseline.json"):
    out = json.load(open("analyses/xgboost_baseline.json"))
for name, fn in PANEL_LOADERS.items():
    if name in out:
        continue
    t0 = time.time()
    try:
        ds = fn()
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                            subsample=0.9, colsample_bytree=0.9,
                            eval_metric="logloss", nthread=2, random_state=0)
        p = cross_val_predict(clf, ds.X, ds.y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(ds.y, p)
        # compare to best committed deep/classical model
        panel_path = f"results/panel_{name}.json"
        best = None
        if os.path.exists(panel_path):
            r = json.load(open(panel_path))
            b = max(r["models"].items(),
                    key=lambda kv: kv[1]["summary"]["roc_auc"]["mean"])
            best = {"model": b[0], "auc": b[1]["summary"]["roc_auc"]["mean"]}
        out[name] = {"tool": "xgboost", "auc": float(auc), "n": int(len(ds.y)),
                     "d": int(ds.X.shape[1]), "best_committed": best,
                     "xgb_wins": bool(best and auc > best["auc"])}
        json.dump(out, open("analyses/xgboost_baseline.json", "w"), indent=1)
        print(f"DONE {name} {time.time()-t0:.0f}s AUC {auc:.4f}", flush=True)
    except Exception as e:
        print(f"FAIL {name} {type(e).__name__}: {str(e)[:80]}", flush=True)
        out[name] = {"tool": "xgboost", "error": str(e)[:120]}
json.dump(out, open("analyses/xgboost_baseline.json", "w"), indent=1)
print("XGB_DONE", flush=True)
