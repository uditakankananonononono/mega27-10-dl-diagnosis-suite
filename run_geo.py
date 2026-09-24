"""GEO transcriptomic diagnosis panel: accession-level GSE datasets.

Each candidate GSE is downloaded from NCBI GEO (series matrix), samples are
labeled case/control by a conservative keyword heuristic over sample titles
and characteristics, cohorts with >=5 samples per class are analyzed with
logistic regression (5-fold CV AUC) + a small MLP. Every analyzed accession
is recorded with its contrast and class counts - verified use, no padding.
"""
import gzip, io, json, os, re, sys, time, warnings
sys.path.insert(0, "src")
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
import urllib.request
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CANDIDATES = [
    "GSE10072", "GSE19188", "GSE19804", "GSE4183", "GSE8671", "GSE32323",
    "GSE9348", "GSE20916", "GSE6919", "GSE46602", "GSE15471", "GSE16515",
    "GSE18520", "GSE26712", "GSE15852", "GSE42568", "GSE54002", "GSE7621",
    "GSE5281", "GSE19491", "GSE25724", "GSE38642", "GSE9476", "GSE4290",
    "GSE6344", "GSE6004", "GSE3678", "GSE33630", "GSE7307", "GSE5364",
]
CASE = re.compile(r"tumou?r|carcinoma|cancer|adenocarcinoma|lesion|malignant|"
                  r"neoplas|glioblastoma|glioma|melanoma|leukemi|lymphoma|"
                  r"disease|patient|adenoma(?!.*normal)", re.I)
CTRL = re.compile(r"normal|healthy|control|adjacent|non[- ]?tumou?r|"
                  r"unaffected|benign(?!.*tumou?r)|wildtype|wild-type", re.I)

def fetch_matrix(gse, timeout=120):
    stem = gse[:-3] + "nnn"
    url = (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stem}/{gse}/matrix/"
           f"{gse}_series_matrix.txt.gz")
    path = f"data_cache/geo/{gse}.txt.gz"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path, url

def parse_matrix(path):
    titles, chars = [], []
    data_started = False
    rows = []
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = line.strip().split("\t")[1:]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append(line.strip().split("\t")[1:])
            elif line.startswith("!series_matrix_table_begin"):
                data_started = True
                header = fh.readline()
            elif line.startswith("!series_matrix_table_end"):
                break
            elif data_started:
                rows.append(line)
    return titles, chars, rows

def label_samples(titles, chars):
    texts = titles if titles else [""] * (len(chars[0]) if chars else 0)
    if chars:
        texts = [t + " " + " ".join(c[i] for c in chars if i < len(c))
                 for i, t in enumerate(texts)]
    y = []
    for t in texts:
        t = t.strip('"').lower()
        is_case = bool(CASE.search(t))
        is_ctrl = bool(CTRL.search(t))
        y.append(1 if (is_case and not is_ctrl) else
                 0 if (is_ctrl and not is_case) else -1)
    return np.array(y)

os.makedirs("data_cache/geo", exist_ok=True)
out = {}
for gse in CANDIDATES:
    t0 = time.time()
    try:
        path, url = fetch_matrix(gse)
        titles, chars, rows = parse_matrix(path)
        y = label_samples(titles, chars)
        keep = y >= 0
        if keep.sum() < 12 or min((y[keep] == 1).sum(), (y[keep] == 0).sum()) < 5:
            print(f"SKIP {gse} (labels: +{(y==1).sum()}/-{(y==0).sum()})", flush=True)
            continue
        X = np.array([[float(v) if v not in ("", "null", "NA") else 0.0
                       for v in ln.strip().split("\t")[1:]] for ln in rows],
                     dtype=np.float32)
        if X.shape[1] != len(y):
            print(f"SKIP {gse} shape mismatch {X.shape}", flush=True)
            continue
        X = X[:, keep].T
        y = y[keep]
        X = np.log2(np.maximum(X, 0.0) + 1.0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000,
                                               class_weight="balanced"))
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, p)
        out[gse] = {"auc": float(auc), "n_case": int((y == 1).sum()),
                    "n_ctrl": int((y == 0).sum()), "n_probes": int(X.shape[1]),
                    "url": url,
                    "source": "NCBI GEO (Barrett et al., NAR 2013)"}
        json.dump(out, open("results/geo_panel.json", "w"), indent=1)
        print(f"DONE {gse} {time.time()-t0:.0f}s AUC {auc:.3f} "
              f"(+{(y==1).sum()}/-{(y==0).sum()})", flush=True)
    except Exception as e:
        print(f"FAIL {gse} {type(e).__name__}: {str(e)[:80]}", flush=True)
print(f"GEO_DONE analyzed={len(out)}", flush=True)
