"""10.8 ClinVar variant-effect model, stage 1: engineered-feature baseline
over the curated germline-SNV subset. Honest leakage control: gene target
encoding computed on train folds only. Compares against the committed
trivial-feature floor (43.2% acc, results/genetic/tool_battery_1_*)."""
import json, re, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders

OUT = Path(__file__).resolve().parent.parent / "results"

AA3 = {"Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His","Ile","Leu","Lys",
       "Met","Phe","Pro","Ser","Thr","Trp","Tyr","Val"}
P_PROT = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter|=)")


def parse_name(name):
    """Extract protein-change features from the HGVS-ish Name field."""
    m = P_PROT.search(name)
    if not m:
        return (0, 0, 0, 0)
    ref, pos, alt = m.group(1), int(m.group(2)), m.group(3)
    missense = int(alt not in ("=", "Ter"))
    nonsense = int(alt == "Ter")
    syn = int(alt == "=")
    return (missense, nonsense, syn, min(pos, 5000))


def build(cap=400000, seed=42):
    t0 = time.time()
    feats, labels, genes = [], [], []
    seen = set()
    for r in loaders.iter_clinvar_subset():
        vid = r["variation_id"]
        if vid in seen:
            continue
        seen.add(vid)
        base = {"A": 0, "C": 1, "G": 2, "T": 3}.get
        ref, alt = base(r["ref"][:1], 4), base(r["alt"][:1], 4)
        ti = int((ref, alt) in ((0, 2), (2, 0), (1, 3), (3, 1)))  # transition
        missense, nonsense, syn, pos = parse_name(r["name"])
        chrom = r["chrom"] if r["chrom"].isdigit() else {"X": 23, "Y": 24}.get(r["chrom"], 25)
        feats.append([int(chrom), ref, alt, ti, missense, nonsense, syn, pos])
        labels.append(r["label"])
        genes.append(r["gene"])
        if len(labels) >= cap:
            break
    X = np.array(feats, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    genes = np.array(genes)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    k = int(0.8 * len(y))
    tr, te = perm[:k], perm[k:]
    # leakage-safe gene target encoding: pathogenic rate per gene on TRAIN only
    rates, global_rate = {}, float(y[tr].mean())
    for g in np.unique(genes[tr]):
        mask = genes[tr] == g
        n = int(mask.sum())
        rates[g] = (y[tr][mask].sum() + 10 * global_rate) / (n + 10)  # smoothed
    te_rate = np.array([rates.get(g, global_rate) for g in genes[te]], dtype=np.float32)
    tr_rate = np.array([rates[g] for g in genes[tr]], dtype=np.float32)
    Xtr = np.column_stack([X[tr], tr_rate])
    Xte = np.column_stack([X[te], te_rate])
    print(f"build {time.time()-t0:.0f}s n={len(y)} uniq_genes={len(rates)}", flush=True)
    return Xtr, y[tr], Xte, y[te]


def train(cap=400000):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
    from statsmodels.stats.proportion import proportion_confint
    Xtr, ytr, Xte, yte = build(cap)
    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=None)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pred = (p > 0.5).astype(int)
    acc = float(accuracy_score(yte, pred))
    auc = float(roc_auc_score(yte, p))
    bacc = float(balanced_accuracy_score(yte, pred))
    lo, hi = proportion_confint(int(acc * len(yte)), len(yte), alpha=0.05, method="wilson")
    np.savez(OUT / "genetic" / "clinvar_gbc_probs.npz", y_true=yte, prob_pos=p)
    json.dump({"dataset": "clinvar_subset (unique VariationIDs)",
               "model": "sklearn HistGradientBoosting on HGVS-parse + locus + leakage-safe gene target encoding",
               "n_train": int(len(ytr)), "n_test": int(len(yte)),
               "test_acc": round(acc, 4), "test_auc": round(auc, 4), "test_bacc": round(bacc, 4),
               "wilson95": [round(float(lo), 4), round(float(hi), 4)],
               "prevalence_pathogenic": round(float(yte.mean()), 4),
               "vs_floor": "trivial locus/allele floor was 43.2% acc (tool_battery_1)",
               "tools": ["scikit-learn", "statsmodels", "numpy", "re"],
               "train_seconds": round(time.time() - t0, 1)},
              open(OUT / "genetic" / "clinvar_gbc_results.json", "w"), indent=1)
    print("CLINVAR GBC DONE", acc, auc, flush=True)


if __name__ == "__main__":
    train(int(sys.argv[1]) if len(sys.argv) > 1 else 400000)
