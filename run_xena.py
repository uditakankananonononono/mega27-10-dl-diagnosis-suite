"""UCSC Xena / GDC TCGA arm: per-cohort primary-tumor vs solid-tissue-normal.

One accession-level dataset per TCGA cohort (TCGA-BRCA, ...). Expression
matrices are the GDC hub STAR count tables (log2 CPM+1 normalized); labels come from the TCGA
barcode sample-type code (positions 14-15): 01-09 = tumor (case), 10-19 =
normal (control); metastatic (06) and all other codes are discarded to keep
the contrast clean. Min 5 per class, else SKIP. Same eval as the GEO arm:
5-fold stratified CV, StandardScaler + balanced logistic regression, AUC.
Size guard: top-variance gene subsample above 40M elements.
"""
import gzip, io, json, os, sys, time, warnings
import urllib.request
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

COHORTS = ("ACC BLCA BRCA CESC CHOL COAD DLBC ESCA GBM HNSC KICH KIRC KIRP "
           "LAML LGG LIHC LUAD LUSC MESO OV PAAD PCPG PRAD READ SARC SKCM "
           "STAD TGCT THCA THYM UCEC UCS UVM").split()

os.makedirs("data_cache/xena", exist_ok=True)
out = {}
if os.path.exists("results/xena_panel.json"):
    out = json.load(open("results/xena_panel.json"))
fail_path = "results/xena_tried.json"
failed = json.load(open(fail_path)) if os.path.exists(fail_path) else []
tried = set(out) | set(failed)

for co in COHORTS:
    acc = f"TCGA-{co}"
    if acc in tried:
        continue
    t0 = time.time()
    url = f"https://gdc.xenahubs.net/download/{acc}.star_counts.tsv.gz"
    path = f"data_cache/xena/{acc}.tsv.gz"
    try:
        if not os.path.exists(path):
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(path, "wb") as fh:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    fh.write(chunk)
        hdr = None
        genes, rows = [], []
        with gzip.open(path, "rt", errors="replace") as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            for line in fh:
                rows.append(line)
        samples = hdr[1:]
        codes = []
        for s in samples:
            parts = s.split("-")
            code = parts[3][:2] if len(parts) > 3 else "??"
            codes.append(code)
        codes = np.array(codes)
        keep = np.array([c in {f"0{i}" for i in range(1, 10)} or
                         (c.startswith("1") and c.isdigit()) for c in codes])
        y = np.array([1 if c[0] == "0" else 0 for c in codes[keep]])
        if keep.sum() < 12 or (y == 1).sum() < 5 or (y == 0).sum() < 5:
            print(f"SKIP {acc} (+{(y==1).sum()}/-{(y==0).sum()})", flush=True)
            failed.append(acc)
            json.dump(sorted(set(failed)), open(fail_path, "w"))
            continue
        idx = np.where(keep)[0]
        X = np.empty((len(rows), int(keep.sum())), dtype=np.float32)
        bad = 0
        for i, ln in enumerate(rows):
            vals = ln.rstrip("\n").split("\t")
            if len(vals) != len(hdr):
                bad += 1
                continue
            genes.append(vals[0])
            X[len(genes) - 1] = [float(vals[j + 1]) for j in idx]
        X = X[:len(genes)].T  # samples x genes (raw STAR counts)
        lib = np.maximum(X.sum(axis=1, keepdims=True), 1.0)
        X = np.log2(X / lib * 1e6 + 1.0)  # log2(CPM+1)
        if X.size > 40_000_000:
            keep_p = max(2000, 40_000_000 // X.shape[0])
            v = np.var(X, axis=0)
            X = X[:, np.argsort(v)[-keep_p:]]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000,
                                               class_weight="balanced"))
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, p)
        out[acc] = {"auc": float(auc), "n_case": int((y == 1).sum()),
                    "n_ctrl": int((y == 0).sum()), "n_genes": int(X.shape[1]),
                    "labeling": "barcode sample-type code (01-09 tumor, 10-19 normal)",
                    "url": url,
                    "source": "UCSC Xena GDC hub (Goldman et al., Nat Biotechnol 2020)"}
        json.dump(out, open("results/xena_panel.json", "w"), indent=1)
        del X
        print(f"DONE {acc} {time.time()-t0:.0f}s AUC {auc:.3f} "
              f"(+{(y==1).sum()}/-{(y==0).sum()})", flush=True)
    except Exception as e:
        print(f"FAIL {acc} {type(e).__name__}: {str(e)[:80]}", flush=True)
        failed.append(acc)
    json.dump(sorted(set(failed)), open(fail_path, "w"))
print(f"XENA_DONE analyzed_total={len(out)}", flush=True)
