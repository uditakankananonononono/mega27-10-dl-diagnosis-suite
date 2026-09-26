"""Battery 5 (post-CNN/GBC deepening): shap permutation importance on the
ClinVar GBC, imagehash cross-split near-duplicate leakage audits (PCam,
neuro), networkx gene-consequence graph on ClinVar, biopython IUPAC/ti-tv
audit on ClinVar alleles, plotly interactive metrics dashboard."""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
NP = Path("/home/sandbox/mega27-expansion/np")

FEATS = ["chrom", "ref", "alt", "transition", "missense", "nonsense",
         "synonymous", "prot_pos", "gene_target_enc"]


def shap_gbc():
    import shap
    from sklearn.ensemble import HistGradientBoostingClassifier
    sys.path.insert(0, str(ROOT / "src"))
    import clinvar_model
    Xtr, ytr, Xte, yte = clinvar_model.build(cap=200000)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08)
    clf.fit(Xtr, ytr)
    rng = np.random.RandomState(0)
    sub = rng.choice(len(yte), 500, replace=False)
    Xs = Xte[sub]
    ex = shap.explainers.Permutation(clf.predict_proba, Xs, max_evals=2 * Xs.shape[1] + 1)
    sv = ex(Xs)
    imp = np.abs(sv.values[..., 1]).mean(axis=0)
    order = np.argsort(-imp)
    rows = [{"feature": FEATS[i], "mean_abs_shap": round(float(imp[i]), 5)} for i in order]
    json.dump({"tool": "shap 0.49 PermutationExplainer",
               "dataset": "clinvar_subset GBC (retrained, same recipe as clinvar_model.py)",
               "n_explain_rows": 500, "class_explained": "pathogenic probability",
               "ranking": rows,
               "note": "permutation importance on 500 held-out rows; ranking, not causal attribution"},
              open(OUT / "genetic" / "clinvar_shap_importance.json", "w"), indent=1)


def _hashes(X, cap=3000):
    import imagehash
    from PIL import Image
    hs = []
    for i in range(min(cap, len(X))):
        arr = X[i]
        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # channel-first cache
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        packed = np.packbits(imagehash.phash(Image.fromarray(arr)).hash).view(np.uint64)
        hs.append(packed)
    return np.array(hs)


def _audit(Xa, Xb, name, out_path, dist=5):
    t0 = time.time()
    ha, hb = _hashes(Xa), _hashes(Xb)
    best = np.full(len(ha), 65)
    pair = None
    for j in range(0, len(hb), 512):
        chunk = hb[j:j + 512]
        d = (ha[:, None] ^ chunk[None, :])
        d = np.unpackbits(d.view(np.uint8), axis=2).sum(axis=2)
        m = d.min(axis=1)
        idx = d.argmin(axis=1)
        upd = m < best
        best[upd] = m[upd]
        if pair is None or m[upd[:len(m)]].any():
            i0 = int(np.argmin(m))
            if pair is None or m[i0] < pair[2]:
                pair = (i0, int(idx[i0] + j), int(m[i0]))
    n_dup = int((best <= dist).sum())
    json.dump({"tool": "imagehash (phash, 64-bit)",
               "dataset": name,
               "method": f"cross-split min Hamming distance, duplicate threshold <= {dist}",
               "n_a": int(len(ha)), "n_b": int(len(hb)),
               "n_a_with_near_dup_in_b": n_dup,
               "near_dup_rate_pct": round(100 * n_dup / len(ha), 2),
               "closest_pair": {"a_index": pair[0], "b_index": pair[1], "hamming": pair[2]},
               "elapsed_s": round(time.time() - t0, 1),
               "note": "cross-split near-duplicates would indicate leakage; rate reported, not hidden"},
              open(out_path, "w"), indent=1)


def imagehash_pcam():
    dv = np.load(NP / "pcam_valid_6144.npz"); dt = np.load(NP / "pcam_test_4096.npz")
    _audit(dv["X"], dt["X"], "pcam valid vs test", OUT / "cancer" / "pcam_imagehash_leakage.json")


def imagehash_neuro():
    d = np.load(NP / "neuro_64.npz")
    _audit(d["Xtr"], d["Xte"], "neuro train vs test", OUT / "neuro" / "neuro_imagehash_leakage.json")


def networkx_clinvar():
    import networkx as nx
    from collections import Counter
    sys.path.insert(0, str(ROOT / "src"))
    import clinvar_model, loaders
    gene_hits, edge_ct = Counter(), Counter()
    seen = set()
    for r in loaders.iter_clinvar_subset():
        vid = r["variation_id"]
        if vid in seen:
            continue
        seen.add(vid)
        m = clinvar_model.P_PROT.search(r["name"])
        kind = ("nonsense" if m and m.group(3) == "Ter" else
                "synonymous" if m and m.group(3) == "=" else
                "missense" if m else "other")
        g = r["gene"] or "unknown"
        gene_hits[g] += 1
        edge_ct[(g, kind)] += 1
        if len(seen) >= 400000:
            break
    top = {g for g, _ in gene_hits.most_common(30)}
    G = nx.Graph()
    for (g, k), c in edge_ct.items():
        if g in top:
            G.add_edge(g, k, weight=c)
    dens = nx.density(G)
    deg = sorted(((n, d) for n, d in G.degree() if n in top), key=lambda x: -x[1])[:10]
    json.dump({"tool": "networkx",
               "dataset": "clinvar_subset, top-30 genes x consequence types",
               "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
               "density": round(dens, 4),
               "top_gene_degrees": [{"gene": n, "degree": int(d)} for n, d in deg],
               "edge_weights_sample": [{"gene": g, "consequence": k, "variants": c}
                                       for (g, k), c in edge_ct.most_common(10)],
               "note": "bipartite-ish gene-consequence co-occurrence over 400k unique VariationIDs"},
              open(OUT / "genetic" / "clinvar_networkx_graph.json", "w"), indent=1)


def biopython_hgvs():
    from Bio.Seq import Seq
    from Bio.Data import IUPACData
    sys.path.insert(0, str(ROOT / "src"))
    import loaders
    valid = set(IUPACData.unambiguous_dna_letters)
    n = ti = tv = bad = 0
    seen = set()
    for r in loaders.iter_clinvar_subset():
        vid = r["variation_id"]
        if vid in seen:
            continue
        seen.add(vid)
        ref, alt = r["ref"][:1].upper(), r["alt"][:1].upper()
        if ref not in valid or alt not in valid:
            bad += 1
            continue
        s_ref, s_alt = Seq(ref), Seq(alt)
        pur = {"A", "G"}
        if (s_ref[0] in pur) == (s_alt[0] in pur):
            ti += 1
        else:
            tv += 1
        n += 1
        if n >= 400000:
            break
    json.dump({"tool": "biopython (Bio.Seq + IUPACData)",
               "dataset": "clinvar_subset alleles, 400k unique VariationIDs",
               "n_checked": n, "transitions": ti, "transversions": tv,
               "ti_tv_ratio": round(ti / max(tv, 1), 3),
               "non_iupac_first_base_skipped": bad,
               "expected_human_ti_tv": "~2.0-2.1 genome-wide (literature anchor)",
               "note": "transition = purine->purine or pyrimidine->pyrimidine via Bio.Seq bases"},
              open(OUT / "genetic" / "clinvar_biopython_titv.json", "w"), indent=1)


def plotly_dashboard():
    import plotly.graph_objects as go
    rows = []
    for f, suite, metric_keys in [
        ("cancer/pcam_cnn_train.json", "PCam CNN", ("acc", "auc")),
        ("cancer/breakhis_cnn_train.json", "BreakHis CNN", ("acc", "auc")),
        ("neuro/brain_mri_cnn_train.json", "BrainMRI CNN (leaky mirror)", ("acc",)),
        ("neuro/brain_mri_cnn_dedup_train.json", "BrainMRI CNN DEDUP", ("acc",)),
        ("genetic/clinvar_gbc_results.json", "ClinVar GBC", ("test_acc", "test_auc")),
    ]:
        d = json.load(open(OUT / f))
        if "history" in d:
            fin = d["history"][-1]
            for k in metric_keys:
                rows.append({"suite": suite, "metric": k, "value": fin.get(k)})
        else:
            for k in metric_keys:
                rows.append({"suite": suite, "metric": k, "value": d.get(k)})
    fig = go.Figure()
    for r in rows:
        fig.add_trace(go.Bar(name=f"{r['suite']} {r['metric']}", x=[r["suite"]], y=[r["value"]]))
    fig.update_layout(title="10.6-10.8 suites: final model metrics", barmode="group",
                      yaxis_title="value")
    out_html = OUT / "fig_suite_metrics_dashboard.html"
    fig.write_html(str(out_html))
    json.dump({"tool": "plotly", "figure": "fig_suite_metrics_dashboard.html",
               "rows": rows, "note": "interactive cross-suite metrics dashboard"},
              open(OUT / "tool_battery_5_plotly.json", "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    {"shap": shap_gbc, "imagehash_pcam": imagehash_pcam, "imagehash_neuro": imagehash_neuro,
     "networkx": networkx_clinvar, "biopython": biopython_hgvs,
     "plotly": plotly_dashboard}[which]()
    print(which, "done", flush=True)
