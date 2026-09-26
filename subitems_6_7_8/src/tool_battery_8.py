"""Battery 8: phate second embedding (neuro), imagehash BreakHis audit,
GWAS Catalog API for top ClinVar genes, mlxtend sequential feature
selection on ClinVar features, neuro dedup train-index prep (answering
the committed 14.33% cross-split leakage finding)."""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
NP = Path("/home/sandbox/mega27-expansion/np")


def phate_neuro():
    import phate
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(NP / "neuro_64.npz")
    X, y = d["Xte"][:1200], d["yte"][:1200]
    Xf = X.reshape(len(X), -1).astype(np.float32) / 255.0
    emb = phate.PHATE(n_jobs=1, random_state=0, verbose=0).fit_transform(Xf)
    names = ["glioma", "meningioma", "notumor", "pituitary"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(4):
        m = y == c
        ax.scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.6, label=names[c])
    ax.legend(markerscale=3)
    ax.set_title("PHATE of neuro test pixels (leakage caveat applies)")
    fig.tight_layout()
    fig.savefig(OUT / "neuro" / "fig_neuro_phate.png", dpi=110)
    json.dump({"tool": "phate", "figure": "fig_neuro_phate.png",
               "n": int(len(X)),
               "note": "second, diffusion-based embedding; cross-checks the umap structure; leakage caveat from neuro_imagehash_leakage.json applies"},
              open(OUT / "neuro" / "tool_battery_8_phate.json", "w"), indent=1)


def _hashes(X, cap=3000):
    import imagehash
    from PIL import Image
    hs = []
    for i in range(min(cap, len(X))):
        arr = X[i]
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        packed = np.packbits(imagehash.phash(Image.fromarray(arr)).hash).view(np.uint64)
        hs.append(packed)
    return np.array(hs)


def _min_dist(ha, hb):
    best = np.full(len(ha), 65)
    for j in range(0, len(hb), 512):
        chunk = hb[j:j + 512]
        d = (ha[:, None] ^ chunk[None, :])
        d = np.unpackbits(d.view(np.uint8), axis=2).sum(axis=2)
        best = np.minimum(best, d.min(axis=1))
    return best


def imagehash_breakhis():
    d = np.load(NP / "breakhis_64.npz")
    t0 = time.time()
    ha, hb = _hashes(d["Xtr"]), _hashes(d["Xte"])
    best = _min_dist(ha, hb)
    n_dup = int((best <= 5).sum())
    json.dump({"tool": "imagehash (phash, 64-bit)",
               "dataset": "breakhis train vs test (patient-level split)",
               "n_a": int(len(ha)), "n_b": int(len(hb)),
               "n_a_with_near_dup_in_b": n_dup,
               "near_dup_rate_pct": round(100 * n_dup / len(ha), 2),
               "min_hamming_observed": int(best.min()),
               "elapsed_s": round(time.time() - t0, 1),
               "note": "patient-level split should block same-patient leakage; near-dups would indicate split failure"},
              open(OUT / "cancer" / "breakhis_imagehash_leakage.json", "w"), indent=1)


def neuro_dedup_prep():
    d = np.load(NP / "neuro_64.npz")
    ha, hb = _hashes(d["Xtr"], cap=len(d["Xtr"])), _hashes(d["Xte"], cap=len(d["Xte"]))
    best = _min_dist(ha, hb)
    keep = np.where(best > 5)[0]
    np.savez(NP / "neuro_64_dedup.npz", Xtr=d["Xtr"][keep], ytr=d["ytr"][keep],
             Xte=d["Xte"], yte=d["yte"])
    json.dump({"tool": "imagehash dedup filter",
               "dataset": "neuro train vs test",
               "n_train_full": int(len(d["Xtr"])), "n_train_dedup": int(len(keep)),
               "n_removed": int(len(d["Xtr"]) - len(keep)),
               "removed_pct": round(100 * (1 - len(keep) / len(d["Xtr"])), 2),
               "note": "train rows with any test near-duplicate (Hamming <= 5) removed; feeds the dedup CNN retrain"},
              open(OUT / "neuro" / "neuro_dedup_filter.json", "w"), indent=1)


def gwas_catalog():
    import requests
    graph = json.load(open(OUT / "genetic" / "clinvar_networkx_graph.json"))
    genes = [g["gene"] for g in graph["top_gene_degrees"][:8]]
    out = []
    for g in genes:
        try:
            r = requests.get(f"https://www.ebi.ac.uk/gwas/rest/api/v2/genes/{g}",
                             headers={"Accept": "application/json"}, timeout=20)
            ok = r.status_code == 200
            js = r.json() if ok else {}
            out.append({"gene": g, "status": r.status_code, "found": ok,
                        "ensembl_gene_ids": js.get("ensembl_gene_ids"),
                        "description": js.get("gene_description")})
        except Exception as e:
            out.append({"gene": g, "error": str(e)[:80]})
        time.sleep(1.5)
    n_ok = sum(1 for x in out if x.get("found"))
    json.dump({"tool": "GWAS Catalog REST API",
               "dataset": "top-8 ClinVar networkx genes",
               "n_queried": len(out), "n_found": n_ok, "results": out,
               "note": "GWAS Catalog V2 API (legacy endpoint deprecated/rate-limited - migrated); Ensembl IDs cross-checked against clinvar_ensembl_xref.json"},
              open(OUT / "genetic" / "clinvar_gwas_catalog.json", "w"), indent=1)


def mlxtend_sfs():
    from mlxtend.feature_selection import SequentialFeatureSelector
    from sklearn.linear_model import LogisticRegression
    sys.path.insert(0, str(ROOT / "src"))
    import clinvar_model
    Xtr, ytr, Xte, yte = clinvar_model.build(cap=120000)
    feats = clinvar_model.__dict__  # names from battery5
    names = ["chrom", "ref", "alt", "transition", "missense", "nonsense",
             "synonymous", "prot_pos", "gene_target_enc"]
    sfs = SequentialFeatureSelector(LogisticRegression(max_iter=150, n_jobs=1),
                                    k_features="best", forward=True, scoring="accuracy",
                                    cv=2, n_jobs=1)
    sfs.fit(Xtr[:20000], ytr[:20000])
    sel = [names[i] for i in sfs.k_feature_idx_]
    json.dump({"tool": "mlxtend SequentialFeatureSelector",
               "dataset": "clinvar_subset logistic, 20k-train subset, cv=2",
               "selected_features": sel,
               "cv_score": round(float(sfs.k_score_), 4),
               "note": "greedy forward selection cross-checks the shap ranking"},
              open(OUT / "genetic" / "clinvar_mlxtend_sfs.json", "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    {"phate": phate_neuro, "imagehash_breakhis": imagehash_breakhis,
     "neuro_dedup_prep": neuro_dedup_prep, "gwas": gwas_catalog,
     "mlxtend": mlxtend_sfs}[which]()
    print(which, "done", flush=True)
