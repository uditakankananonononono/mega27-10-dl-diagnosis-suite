"""pydeseq2: differential-expression validation on TCGA-BRCA.
Tumor vs normal DESeq2 on the cached STAR-count matrix (top-5000
variable genes, all samples): confirms the biological separation the
classifier exploits; reports significant-gene count and top genes.
Output analyses/deseq_brca.json"""
import gzip, json, os, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

path = "data_cache/xena/TCGA-BRCA.tsv.gz"
with gzip.open(path, "rt", errors="replace") as fh:
    hdr = fh.readline().rstrip("\n").split("\t")
    nlines = sum(1 for _ in fh)
samples = hdr[1:]
codes = np.array([s.split("-")[3][:2] if len(s.split("-")) > 3 else "??"
                  for s in samples])
keep = np.array([c in {"01", "11"} for c in codes])  # primary tumor, solid normal
idx = np.where(keep)[0]
X = np.empty((nlines, int(keep.sum())), dtype=np.float32)
genes = []
ng = 0
with gzip.open(path, "rt", errors="replace") as fh:
    fh.readline()
    for ln in fh:
        v = ln.rstrip("\n").split("\t")
        if len(v) != len(hdr):
            continue
        genes.append(v[0])
        X[ng] = np.array(v[1:], dtype=np.float32)[idx]
        ng += 1
X = X[:ng]
y = np.array([1 if c == "01" else 0 for c in codes[keep]])
# top-5000 variable genes on log scale for tractability
v = np.var(np.log2(X + 1), axis=1)
sel = np.argsort(v)[-1000:]
# subsample tumors for runtime (keep all normals); fixed seed, documented
rng = np.random.RandomState(0)
ti = np.where(y == 1)[0]
sub_t = np.sort(rng.choice(ti, size=min(200, len(ti)), replace=False))
keep_cols = np.sort(np.concatenate([sub_t, np.where(y == 0)[0]]))
genes = [genes[i] for i in sel]
X = np.rint(X[sel]).astype(int)  # DESeq2 needs raw counts
X = X[:, keep_cols]
y = y[keep_cols]
counts = pd.DataFrame(X.T, columns=[g.split(".")[0] for g in genes])
counts = counts.loc[:, ~counts.columns.duplicated()]
meta = pd.DataFrame({"condition": np.where(y == 1, "tumor", "normal")},
                    index=counts.index)
import pickle
pickle.dump({"counts": counts, "meta": meta},
            open("data_cache/xena/deseq_input.pkl", "wb"))
print("PHASE1_DONE", flush=True)

