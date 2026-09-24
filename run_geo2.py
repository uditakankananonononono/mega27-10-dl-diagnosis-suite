"""GEO expansion pass 2: eutils-targeted candidates + contrast-discovery labeling.

Pass 1 (run_geo.py) used a conservative keyword heuristic over combined sample
text; its candidate tail yielded ~0 because many cohorts carry labels only in
structured characteristics fields. This pass (a) targets new GSEs via NCBI
eutils esearch (tumor+normal human array studies), and (b) discovers the
case/control contrast from the characteristics fields themselves: a field with
exactly 2 distinct values covering all samples becomes the label iff one value
matches case-only keywords and the other control-only keywords. Conflicts and
ambiguities discard the sample/study. Every analyzed accession is recorded.
"""
import gzip, io, json, os, re, sys, time, warnings
import urllib.request, urllib.parse, xml.etree.ElementTree as ET
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CASE = re.compile(r"tumou?r|carcinoma|cancer|adenocarcinoma|lesion|malignant|"
                  r"neoplas|glioblastoma|glioma|melanoma|leukemi|lymphoma|"
                  r"disease|patient|adenoma(?!.*normal)|metasta", re.I)
CTRL = re.compile(r"normal|healthy|control|adjacent|non[- ]?tumou?r|unaffected|"
                  r"benign(?!.*tumou?r)|wildtype|wild-type|non[- ]?neoplas|"
                  r"peritumou?ral|uninvolved|disease[- ]?free", re.I)

def esearch(term, retmax=250):
    q = urllib.parse.urlencode({"db": "gds", "term": term, "retmax": retmax})
    u = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{q}"
    root = ET.fromstring(urllib.request.urlopen(u, timeout=60).read())
    return [e.text for e in root.iter("Id")]

def esummary_gse(ids):
    if not ids: return []
    q = urllib.parse.urlencode({"db": "gds", "id": ",".join(ids)})
    u = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{q}"
    root = ET.fromstring(urllib.request.urlopen(u, timeout=90).read())
    out = []
    for doc in root.iter("DocSum"):
        gse = n = None
        for item in doc.iter("Item"):
            if item.get("Name") == "Accession": gse = item.text
            if item.get("Name") == "n_samples":
                try: n = int(item.text)
                except (TypeError, ValueError): pass
        if gse and gse.startswith("GSE"): out.append((gse, n))
    return out

def fetch_matrix(gse):
    stem = gse[:-3] + "nnn"
    url = (f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stem}/{gse}/matrix/"
           f"{gse}_series_matrix.txt.gz")
    path = f"data_cache/geo/{gse}.txt.gz"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path, url

def parse_matrix(path):
    titles, chars, rows, started = [], [], [], False
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = line.rstrip("\n").split("\t")[1:]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append(line.rstrip("\n").split("\t")[1:])
            elif line.startswith("!series_matrix_table_begin"):
                started = True; fh.readline()
            elif line.startswith("!series_matrix_table_end"):
                break
            elif started:
                rows.append(line)
    return titles, chars, rows

def label_samples(titles, chars):
    n = len(titles) if titles else (len(chars[0]) if chars else 0)
    # 1) contrast discovery: single field, exactly 2 values, full coverage
    for field in chars:
        vals = [v.strip('"').strip().lower() for v in field[:n]]
        key = vals[0].split(":")[0] if ":" in vals[0] else ""
        if key:
            vals = [v.split(":", 1)[1].strip() if ":" in v else v for v in vals]
        uniq = sorted(set(vals))
        if len(uniq) == 2:
            a, b = uniq
            ca, cb = bool(CASE.search(a)), bool(CASE.search(b))
            na, nb = bool(CTRL.search(a)), bool(CTRL.search(b))
            if ca and not na and nb and not cb:
                return np.array([1 if v == a else 0 for v in vals]), f"field:{a}|{b}"
            if cb and not nb and na and not ca:
                return np.array([1 if v == b else 0 for v in vals]), f"field:{a}|{b}"
    # 2) combined-text keyword heuristic (pass-1 behavior)
    texts = titles if titles else [""] * n
    if chars:
        texts = [t + " " + " ".join(c[i] for c in chars if i < len(c))
                 for i, t in enumerate(texts)]
    y = []
    for t in texts:
        t = t.strip('"').lower()
        c, k = bool(CASE.search(t)), bool(CTRL.search(t))
        y.append(1 if (c and not k) else 0 if (k and not c) else -1)
    return np.array(y), "text"

os.makedirs("data_cache/geo", exist_ok=True)
out = {}
if os.path.exists("results/geo_panel.json"):
    out = json.load(open("results/geo_panel.json"))
tried = set(out)
fail_path = "results/geo_tried.json"
if os.path.exists(fail_path):
    tried |= set(json.load(open(fail_path)))
if os.path.exists("logs/geo_hung.txt"):
    tried |= set(open("logs/geo_hung.txt").read().split())

cands = []
if "--esearch" in sys.argv:
    ids = esearch('("expression profiling by array"[DataSet Type]) AND '
                  '((tumor[All Fields] AND normal[All Fields]) OR '
                  '(cancer[All Fields] AND adjacent[All Fields])) AND '
                  '(human[Organism] OR "homo sapiens"[Organism])')
    for gse, n in esummary_gse(ids):
        if gse not in tried and n and 24 <= n <= 1200:
            cands.append(gse)
else:
    for a in sys.argv[1:]:
        cands.append(a)
print(f"candidates: {len(cands)}", flush=True)

failed = json.load(open(fail_path)) if os.path.exists(fail_path) else []
for gse in cands:
    if gse in out or gse in tried:
        continue
    t0 = time.time()
    try:
        path, url = fetch_matrix(gse)
        titles, chars, rows = parse_matrix(path)
        y, how = label_samples(titles, chars)
        keep = y >= 0
        if keep.sum() < 12 or min((y[keep] == 1).sum(), (y[keep] == 0).sum()) < 5:
            print(f"SKIP {gse} {how} (+{(y==1).sum()}/-{(y==0).sum()})", flush=True)
            failed.append(gse); continue
        ncol = len(titles) if titles else len(chars[0])
        rows = [ln for ln in rows if len(ln.strip().split("\t")) == ncol + 1]
        if len(rows) < 500:
            print(f"SKIP {gse} (only {len(rows)} clean rows)", flush=True)
            failed.append(gse); continue
        X = np.array([[float(v) if v not in ("", "null", "NA") else 0.0
                       for v in ln.strip().split("\t")[1:]] for ln in rows],
                     dtype=np.float32)
        X = X[:, keep].T
        y = y[keep]
        X = np.log2(np.maximum(X, 0.0) + 1.0)
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
        out[gse] = {"auc": float(auc), "n_case": int((y == 1).sum()),
                    "n_ctrl": int((y == 0).sum()), "n_probes": int(X.shape[1]),
                    "labeling": how, "url": url,
                    "source": "NCBI GEO (Barrett et al., NAR 2013)"}
        json.dump(out, open("results/geo_panel.json", "w"), indent=1)
        print(f"DONE {gse} {time.time()-t0:.0f}s AUC {auc:.3f} {how} "
              f"(+{(y==1).sum()}/-{(y==0).sum()})", flush=True)
    except Exception as e:
        print(f"FAIL {gse} {type(e).__name__}: {str(e)[:80]}", flush=True)
        failed.append(gse)
    json.dump(sorted(set(failed)), open(fail_path, "w"))
print(f"GEO2_DONE analyzed_total={len(out)} new_candidates={len(cands)}", flush=True)
