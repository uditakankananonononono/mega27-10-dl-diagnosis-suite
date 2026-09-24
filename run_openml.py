"""OpenML clinical/biomedical diagnosis arm: accession-level datasets (OpenML data IDs).

Each curated data ID is a real, identifier-backed record fetched via the OpenML
REST API (sklearn fetch_openml), labeled by its own published target, and
analyzed with the same CV harness as the UCI panel (logreg + RF,
5-fold stratified CV, ROC AUC; macro-OvR for multiclass). Synthetic/simulated
sets (BNG, GAMETES) and re-upload duplicates of the UCI panel are excluded by
curation. Every analyzed ID is recorded with provenance URL.
"""
import json, os, signal, socket, time, warnings
socket.setdefaulttimeout(120)
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CANDIDATES = {
    8: "liver-disorders-bupa", 25: "colic-horse", 163: "lung-cancer-uci",
    171: "primary-tumor", 224: "breast-tumor", 346: "aids", 481: "biomed",
    553: "kidney", 1039: "hiva-agnostic", 1084: "burkitt-lymphoma",
    1086: "ovarian-tumour-ge", 1087: "hepatitis-c-ge", 1088: "various-cancers",
    1101: "lymphoma-2class", 1107: "tumors-ramaswamy",
    1122: "ap-breast-prostate", 1126: "ap-colon-lung", 1128: "ova-breast",
    1130: "ova-lung", 1245: "lung-shedden", 1412: "lung-gse31210",
    1432: "colon-cancer-alon", 1434: "duke-breast-cancer",
    1464: "blood-transfusion", 1465: "breast-tissue", 1466: "cardiotocography",
    1471: "eeg-eye-state", 1498: "sa-heart", 1506: "thoracic-surgery",
    1523: "vertebra-column", 4134: "bioresponse",
    4531: "parkinsons-telemonitoring", 4540: "parkinson-speech",
    4541: "diabetes-130us", 40474: "thyroid-allbp", 40477: "thyroid-allrep",
    40966: "mice-protein", 41430: "diabetic-mellitus",
    42878: "autism-adult", 42893: "acp-lung-cancer", 42895: "acp-breast-cancer",
    42900: "breast-cancer-coimbra", 42901: "caesarian-section",
    42972: "chronic-kidney", 43008: "heart-failure",
    43414: "autism-screening", 43428: "mexico-covid19",
    43657: "cumida-brain", 43658: "cumida-breast",
    43726: "brca-vs-normal", 43439: "appointment-noshow",
    43284: "dream3-phospho",
}
MAX_ELEMS = 40_000_000  # OOM guard: subsample probes above this

def load(did):
    ds = fetch_openml(data_id=did, data_home="data_cache/openml",
                      as_frame=False, parser="auto")
    X, y = ds.data, ds.target
    X = np.asarray(X)
    if X.dtype == object or str(X.dtype).startswith("<U"):
        import pandas as pd
        X = pd.DataFrame(X).apply(lambda c: pd.factorize(c)[0]).values
    X = np.nan_to_num(X.astype(np.float64), nan=0.0)
    return X, np.asarray(y).astype(str)

os.makedirs("results", exist_ok=True)
out = {}
if os.path.exists("results/openml_panel.json"):
    out = json.load(open("results/openml_panel.json"))
class _T(Exception): pass
def _alarm(sig, frm): raise _T()
signal.signal(signal.SIGALRM, _alarm)
for did, name in CANDIDATES.items():
    if str(did) in out:
        continue
    t0 = time.time()
    signal.alarm(360)
    try:
        X, y = load(did)
        if X.size > MAX_ELEMS:
            keep_p = max(1000, MAX_ELEMS // X.shape[0])
            v = np.nanvar(X, axis=0)
            X = X[:, np.argsort(v)[-keep_p:]]
        if X.shape[0] > 20000:
            idx = np.random.RandomState(0).choice(X.shape[0], 20000, replace=False)
            X, y = X[idx], y[idx]
        cls = np.unique(y)
        multi = len(cls) > 2
        if multi and len(cls) > 20:
            print(f"SKIP {did} {name} ({len(cls)} classes)", flush=True); continue
        if not multi:
            u, c = np.unique(y, return_counts=True)
            if min(c) < 5:
                print(f"SKIP {did} {name} (class counts {dict(zip(u.tolist(), c.tolist()))})", flush=True)
                continue
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        aucs = {}
        for mname, clf in [
            ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight="balanced"))),
            ("rf", RandomForestClassifier(120, n_jobs=2, random_state=0, class_weight="balanced")),
        ]:
            try:
                if multi:
                    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
                    aucs[mname] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
                else:
                    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
                    aucs[mname] = float(roc_auc_score(y, p))
            except Exception:
                pass
        if not aucs:
            print(f"FAIL {did} {name} (all models failed)", flush=True); continue
        best = max(aucs, key=aucs.get)
        out[str(did)] = {
            "name": name, "n": int(X.shape[0]), "p": int(X.shape[1]),
            "n_classes": int(len(cls)), "multiclass": bool(multi),
            "aucs": aucs, "best": best, "best_auc": aucs[best],
            "url": f"https://www.openml.org/d/{did}",
            "source": "OpenML (Vanschoren et al., SIGKDD Explor. 2014)"}
        json.dump(out, open("results/openml_panel.json", "w"), indent=1)
        print(f"DONE {did} {name} {time.time()-t0:.0f}s best {best} {aucs[best]:.3f} n={X.shape[0]} p={X.shape[1]}", flush=True)
    except Exception as e:
        print(f"FAIL {did} {name} {type(e).__name__}: {str(e)[:90]}", flush=True)
    finally:
        signal.alarm(0)
print(f"OPENML_DONE analyzed={len(out)}", flush=True)
