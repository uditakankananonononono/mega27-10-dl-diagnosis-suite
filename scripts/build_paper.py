"""Build the MEGA27-10 paper as a Times New Roman DOCX from result JSONs."""
import glob, json, os, sys
sys.path.insert(0, "src")
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def set_times(doc):
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(11)

def h(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.name = "Times New Roman"
    return p

def para(doc, text):
    return doc.add_paragraph(text)

def table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for j, htxt in enumerate(header):
        c = t.rows[0].cells[j]
        c.text = str(htxt)
        for p in c.paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows:
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = str(v)
    return t

def main():
    doc = Document()
    set_times(doc)
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Cross-Disease Graph Diagnosis: A Clinical Benchmark of "
                      "CNN and GNN Architectures and a Falsifiable Test of "
                      "Cross-Disease Patient Bridges")
    r.bold = True; r.font.size = Pt(16); r.font.name = "Times New Roman"
    sub = doc.add_paragraph("MEGA-27 Item 10 - Deep-Learning Diagnosis Suite. "
                            "September 24, 2026.")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    h(doc, "Abstract")
    para(doc, "We benchmark MLP, 1D-CNN, GCN and GAT diagnosis models against "
              "strong classical baselines on real clinical datasets "
              "(accession-level, URLs verified), with bootstrap confidence "
              "intervals and calibration metrics, and we test a falsifiable "
              "discovery claim: cross-disease patient bridges - edges of a "
              "unified patient-similarity graph spanning diseases, built on "
              "disease-agnostic distributional fingerprints - carry "
              "transferable diagnostic signal. All claims are reported with "
              "seed variance; negative results are preserved.")

    sys.path.insert(0, "scripts")
    import paper_content as pc
    h(doc, "Introduction")
    for pgh in pc.INTRO.strip().split("\n\n"):
        para(doc, pgh.strip())
    h(doc, "Methods")
    for pgh in pc.METHODS.strip().split("\n\n"):
        para(doc, pgh.strip())
    h(doc, "Mathematical formulation")
    eqs = [
        "(1) Gaussian kNN kernel, self-tuned bandwidth: A_ij = exp(-||x_i - x_j||^2 / (2 sigma^2)), sigma^2 = median kNN squared distance (Zelnik-Manor & Perona).",
        "(2) Symmetric GCN normalization: A_hat = D~^{-1/2} (A + I) D~^{-1/2} (Kipf & Welling 2017).",
        "(3) GCN layer: H^{l+1} = tanh(A_hat H^l W^l). Derivation: first-order Chebyshev approximation of spectral graph convolution with lambda_max ~= 2 and renormalization trick.",
        "(4) Distributional fingerprint phi(x) = (mu, sigma, min, max, q25, q50, q75, m3, m4, |z|_1/d), m_r = mean z_j^r, z_j = (x_j - mu)/sigma - disease-agnostic by construction.",
        "(5) Class-weighted BCE: L = -(1/n) sum_i w_{y_i}[y_i log s(f_i) + (1 - y_i) log(1 - s(f_i))], w_1 = n_-/n_+.",
        "(6) Multi-task transductive objective: L_MT = sum_d L^{(d)}(T_d); grad L_MT = sum_d grad L^{(d)}.",
        "(7) ECE = sum_b (|S_b|/n) |acc(S_b) - conf(S_b)| over 10 equal-width bins.",
        "(8) Percentile bootstrap CI_95 = [Q_0.025(theta*_b), Q_0.975(theta*_b)], B = 1000 resamples.",
        "(9) Bridge ablation: Delta^{(d)} = (1/S) sum_s (AUC_full - AUC_no-bridge); sign-test p = 2^{-S} sum_{k>=k0} C(S,k) under the no-signal null (Proposition 1).",
        "(10) Mann-Whitney AUC: AUC = P(f(x+) > f(x-)) + 0.5 P(f(x+) = f(x-)).",
        "(11) Bridge homophily h = |{(i,j) in E_cross : y_i = y_j}| / |E_cross|.",
    ]
    for e in eqs:
        para(doc, e)

    # ---- datasets + results from JSONs ----
    h(doc, "Datasets")
    rows = []
    for f in sorted(glob.glob("results/*.json")):
        r = json.load(open(f))
        if "dataset" in r:
            rows.append([r["dataset"], r["n"], r["d"], r.get("citation", "")])
    if rows:
        table(doc, ["Dataset", "n", "features", "Citation"], rows)

    h(doc, "Results")
    para(doc, "Each subsection reports the committed result file for one "
              "dataset: per-model ROC AUC (mean +- std across seeds), "
              "balanced accuracy, Brier score and ECE. The best model is "
              "named; all numbers trace to results/*.json in the repository.")
    for f in sorted(glob.glob("results/*.json")):
        r = json.load(open(f))
        if "models" in r:
            h(doc, f"Dataset: {r['dataset']}", level=2)
            rows = []
            for m, b in r["models"].items():
                s = b["summary"]
                rows.append([m,
                             f"{s['roc_auc']['mean']:.4f} +- {s['roc_auc']['std']:.4f}",
                             f"{s['balanced_accuracy']['mean']:.4f}",
                             f"{s['brier']['mean']:.4f}",
                             f"{s['ece']['mean']:.4f}"])
            table(doc, ["Model", "ROC AUC", "Balanced acc", "Brier", "ECE"], rows)
        elif "aggregate" in r:
            h(doc, "Discovery experiment: cross-disease bridges", level=2)
            rows = []
            for name, a in r["aggregate"].items():
                rows.append([name, f"{a['full_auc_mean']:.4f}",
                             f"{a['ablated_auc_mean']:.4f}",
                             f"{a['bridge_delta_auc_mean']:+.4f}",
                             str(a["bridge_delta_auc_per_seed"])])
            table(doc, ["Disease", "AUC full", "AUC ablated", "Delta",
                        "Delta per seed"], rows)
            if r.get("stable_bridges"):
                para(doc, "Stable bridges (seed recurrence >= 4/5): "
                          + json.dumps(r["stable_bridges"]))

    h(doc, "External tools")
    tools = json.load(open("paper/tools_manifest.json")) if \
        os.path.exists("paper/tools_manifest.json") else []
    if tools:
        table(doc, ["Tool", "Version", "Used for"],
              [[t["tool"], t.get("version", ""), t["used_for"]] for t in tools])

    h(doc, "Related work")
    for pgh in pc.RELATED.strip().split("\n\n"):
        para(doc, pgh.strip())
    h(doc, "Protocol and reproducibility")
    for pgh in pc.PROTOCOL.strip().split("\n\n"):
        para(doc, pgh.strip())
    h(doc, "Mathematical derivations")
    for pgh in pc.DERIVATIONS.strip().split("\n\n"):
        para(doc, pgh.strip())
    h(doc, "Discovery experiment: cross-disease bridges")
    for pgh in pc.DISCOVERY.strip().split("\n\n"):
        para(doc, pgh.strip())
    h(doc, "Label efficiency")
    for pgh in pc.LABEL_EFF.strip().split("\n\n"):
        para(doc, pgh.strip())
    if os.path.exists("results/label_efficiency.json"):
        le = json.load(open("results/label_efficiency.json"))
        rows = []
        for name in le:
            for f in ("0.1", "0.25", "0.5", "1.0"):
                g, m = le[name]["gcn"].get(f), le[name]["mlp"].get(f)
                if g and m:
                    import numpy as np
                    rows.append([name, f"{float(f)*100:.0f}%",
                                 f"{np.mean(g):.4f}", f"{np.mean(m):.4f}"])
        table(doc, ["Dataset", "Labels", "GCN AUC", "MLP AUC"], rows)
    h(doc, "Published SOTA comparison")
    if os.path.exists("results/sota_comparison.json"):
        sc = json.load(open("results/sota_comparison.json"))
        para(doc, sc.get("note", ""))
        rows = []
        for ds, claims in sc.items():
            if isinstance(claims, list):
                for c in claims:
                    rows.append([ds, c["claim"], c["source"]])
        table(doc, ["Dataset", "Published claim", "Source"], rows)
    h(doc, "Transcriptomic panel (GEO)")
    for pgh in pc.GEO_ARM.strip().split("\n\n"):
        para(doc, pgh.strip())
    if os.path.exists("results/geo_panel.json"):
        geo = json.load(open("results/geo_panel.json"))
        rows = [[g, f"{v['auc']:.3f}", f"+{v['n_case']}/-{v['n_ctrl']}",
                 v["n_probes"]] for g, v in sorted(geo.items())]
        table(doc, ["GEO accession", "CV AUC", "cases/controls", "probes"],
              rows)
        h(doc, "Per-accession notes", level=2)
        for g, v in sorted(geo.items()):
            para(doc,
                 f"{g}: {v['n_case']} case and {v['n_ctrl']} control samples "
                 f"retained after ambiguous-label discard, {v['n_probes']} "
                 f"probe features; labeling by {v['labeling']}; five-fold CV "
                 f"AUC {v['auc']:.3f}. Source matrix: {v['url']}.")
    h(doc, "Medical image arm (MedMNIST)")
    for pgh in pc.MEDMNIST_ARM.strip().split("\n\n"):
        para(doc, pgh.strip())
    if os.path.exists("results/medmnist.json"):
        mm = json.load(open("results/medmnist.json"))
        rows = [[k, v["task"], f"{v['auc']:.4f}", f"{v['n_train']}/{v['n_test']}"]
                for k, v in sorted(mm.items())]
        table(doc, ["Subset", "Task", "CNN AUC", "n train/test"], rows)
    h(doc, "OpenML broad panel")
    for pgh in pc.OPENML_ARM.strip().split("\n\n"):
        para(doc, pgh.strip())
    if os.path.exists("results/openml_panel.json"):
        om = json.load(open("results/openml_panel.json"))
        rows = [[v.get("name", k), v["n"], v["p"],
                 v.get("n_classes", 2), v.get("best", ""),
                 f"{v.get('best_auc', 0):.4f}", v.get("url", "")]
                for k, v in sorted(om.items(), key=lambda kv: kv[1].get("name", ""))]
        table(doc, ["Dataset", "n", "features", "classes", "best model",
                    "best AUC", "URL"], rows)
    if os.path.exists("results/openml_failures.json"):
        of = json.load(open("results/openml_failures.json"))
        h(doc, "OpenML candidate failures (documented)", level=2)
        if isinstance(of, dict):
            rows = [[v.get("name", k) if isinstance(v, dict) else k,
                     v.get("reason", "") if isinstance(v, dict) else str(v),
                     v.get("url", "") if isinstance(v, dict) else ""]
                    for k, v in of.items()]
        else:
            rows = [[str(x), "", ""] for x in of]
        table(doc, ["Candidate", "Reason", "URL"], rows)

    h(doc, "Pan-cancer arm (TCGA via UCSC Xena)")
    for pgh in pc.XENA_ARM.strip().split("\n\n"):
        para(doc, pgh.strip())
    if os.path.exists("results/xena_panel.json"):
        xe = json.load(open("results/xena_panel.json"))
        rows = [[k, f"{v['auc']:.4f}", f"+{v['n_case']}/-{v['n_ctrl']}",
                 v["n_genes"]] for k, v in sorted(xe.items())]
        table(doc, ["Cohort", "CV AUC", "tumor/normal", "genes"], rows)
        h(doc, "Per-cohort notes", level=2)
        for k, v in sorted(xe.items()):
            para(doc,
                 f"{k}: {v['n_case']} primary-tumor and {v['n_ctrl']} "
                 f"solid-tissue-normal samples, {v['n_genes']} genes after "
                 f"variance subsampling; five-fold CV AUC {v['auc']:.4f}. "
                 f"Labels from TCGA barcode sample-type codes; matrix: "
                 f"{v['url']}.")
    if os.path.exists("results/xena_tried.json"):
        xt = json.load(open("results/xena_tried.json"))
        para(doc, "Cohorts screened but not analyzed (fewer than five "
                  "normal samples, or retrieval failure): " + ", ".join(xt) + ".")

    h(doc, "Volumetric image arm (MedMNIST 3D)")
    for pgh in pc.MM3D_ARM.strip().split("\n\n"):
        para(doc, pgh.strip())
    if os.path.exists("results/medmnist3d.json"):
        m3 = json.load(open("results/medmnist3d.json"))
        rows = [[k, v["task"], f"{v['auc']:.4f}", f"{v['n_train']}/{v['n_test']}"]
                for k, v in sorted(m3.items())]
        table(doc, ["Subset", "Task", "3D-CNN AUC", "n train/test"], rows)

    h(doc, "Second-wave analyses over committed panels")
    for pgh in pc.NEW_ANALYSES.strip().split("\n\n"):
        para(doc, pgh.strip())
    if os.path.exists("analyses/xgboost_baseline.json"):
        xb = json.load(open("analyses/xgboost_baseline.json"))
        h(doc, "Gradient-boosted baseline (XGBoost) vs committed best", level=2)
        rows = []
        for k, v in sorted(xb.items()):
            if "auc" in v and v.get("best_committed"):
                rows.append([k, f"{v['auc']:.4f}",
                             f"{v['best_committed']['model']}",
                             f"{v['best_committed']['auc']:.4f}",
                             "yes" if v["xgb_wins"] else "no"])
        table(doc, ["Dataset", "XGBoost AUC", "Committed best model",
                    "Committed best AUC", "XGBoost wins"], rows)
    if os.path.exists("analyses/stats_tools.json"):
        st = json.load(open("analyses/stats_tools.json"))
        h(doc, "FDR correction of bridge discoveries", level=2)
        fdr = st.get("statsmodels_fdr", {})
        rows = [[d["dataset"], f"{d['delta_auc']:+.4f}", d["seeds_positive"],
                 f"{d['p_raw']:.4f}", f"{d['p_bh']:.4f}",
                 "yes" if d["sig_fdr05"] else "no"]
                for d in fdr.get("per_disease", [])]
        if rows:
            table(doc, ["Disease", "Bridge delta AUC", "Seeds +",
                        "p (raw)", "p (BH)", "sig at FDR 0.05"], rows)
        mc = st.get("mlxtend_mcnemar", {})
        if "holdout" in mc or "mcnemar_table" in mc:
            h(doc, "McNemar test: GCN vs MLP (cleveland)", level=2)
            para(doc, json.dumps(mc, indent=1)[:600])
        es = st.get("pingouin_effect_sizes", {}).get("effect_sizes", [])
        if es:
            h(doc, "Label-efficiency effect sizes", level=2)
            rows = [[e["dataset"], e["label_fraction"],
                     f"{e['gcn_mean']:.4f}", f"{e['mlp_mean']:.4f}",
                     f"{e['cohens_d']:+.2f}"] for e in es]
            table(doc, ["Dataset", "Labels", "GCN AUC", "MLP AUC",
                        "Cohen's d"], rows)
    if os.path.exists("analyses/shap_smote.json"):
        ss = json.load(open("analyses/shap_smote.json"))
        h(doc, "Feature attribution (SHAP, wdbc)", level=2)
        rows = [[t["feature"], f"{t['mean_abs_shap']:.4f}"]
                for t in ss.get("shap_wdbc", {}).get("top10_features", [])]
        if rows:
            table(doc, ["Feature", "mean |SHAP|"], rows)
        h(doc, "SMOTE vs class weighting", level=2)
        rows = [[r["dataset"], f"{r['minority_fraction']:.3f}",
                 f"{r['auc_class_weight']:.4f}", f"{r['auc_smote']:.4f}"]
                for r in ss.get("imblearn_smote", {}).get("results", [])]
        if rows:
            table(doc, ["Dataset", "Minority frac", "AUC class-weight",
                        "AUC SMOTE"], rows)
    if os.path.exists("analyses/embed_clusters.json"):
        ec = json.load(open("analyses/embed_clusters.json"))
        h(doc, "Fingerprint embedding and clustering (UMAP + HDBSCAN)", level=2)
        para(doc, f"Patients: {ec['n_patients']} across "
                  f"{ec['n_datasets']} datasets; clusters: {ec['n_clusters']} "
                  f"(noise fraction {ec['noise_fraction']:.2f}); adjusted Rand "
                  f"index between clusters and disease identity: "
                  f"{ec['ari_clusters_vs_disease']:.4f}. "
                  + ec.get("interpretation", ""))
    if os.path.exists("analyses/validation_xcheck.json"):
        vx = json.load(open("analyses/validation_xcheck.json"))
        h(doc, "Pipeline cross-validation with independent tools", level=2)
        para(doc, json.dumps(vx.get("torchmetrics_xcheck", {}), indent=1)[:500])
        para(doc, json.dumps(vx.get("geoparse_validation", {}), indent=1)[:500])
    if os.path.exists("analyses/pyg_xcheck.json"):
        px = json.load(open("analyses/pyg_xcheck.json"))
        h(doc, "GCN re-implementation cross-check (PyTorch Geometric)", level=2)
        para(doc, f"PyG GCN AUC {px['auc_pyg_gcn']:.4f} vs diagbench GCN AUC "
                  f"{px['auc_diagbench_gcn']:.4f} on the same cleveland graph "
                  f"and split (absolute difference {px['abs_diff']:.4f}). "
                  + px.get("conclusion", ""))
    if os.path.exists("analyses/deseq_brca.json"):
        dz = json.load(open("analyses/deseq_brca.json"))
        h(doc, "Differential-expression validation (DESeq2, TCGA-BRCA)", level=2)
        para(doc, f"{dz['n_sig_padj05']} of {dz['n_genes_tested']} tested "
                  f"genes significant at padj < 0.05 "
                  f"({dz['n_tumor']} tumor vs {dz['n_normal']} normal). "
                  + dz.get("conclusion", ""))
        rows = [[g["gene"], f"{g['log2fc']:+.2f}", f"{g['padj']:.2e}"]
                for g in dz.get("top20_genes", [])]
        if rows:
            table(doc, ["Gene", "log2 fold-change", "padj"], rows)
    if os.path.exists("analyses/captum_optuna.json"):
        co = json.load(open("analyses/captum_optuna.json"))
        h(doc, "Hyperparameter search bound (optuna, wdbc MLP)", level=2)
        ow = co.get("optuna_wdbc", {})
        if "best_auc" in ow:
            para(doc, f"Best 4-fold CV AUC after 15 TPE trials: "
                      f"{ow['best_auc']:.4f} (committed logistic-regression "
                      f"baseline: {ow.get('committed_logreg_auc')}). "
                      f"Best parameters: {json.dumps(ow.get('best_params', {}))}.")
        cp = co.get("captum_pathmnist", {})
        if "figure" in cp:
            h(doc, "CNN saliency (captum Integrated Gradients)", level=2)
            para(doc, f"Integrated-Gradients attribution on 8 pathmnist test "
                      f"images ({cp['method']}); see figure "
                      f"captum_pathmnist.")

    h(doc, "Negative results")
    for pgh in pc.NEGATIVE.strip().split("\n\n"):
        para(doc, pgh.strip())
    h(doc, "Discussion")
    for pgh in pc.DISCUSSION.strip().split("\n\n"):
        para(doc, pgh.strip())

    h(doc, "Figures")
    for fig in sorted(glob.glob("figures/*.png")):
        doc.add_picture(fig, width=Inches(6))
        para(doc, os.path.basename(fig).replace("_", " ").replace(".png", ""))

    doc.save("paper/MEGA27-10-paper.docx")
    print("saved paper/MEGA27-10-paper.docx")

if __name__ == "__main__":
    main()
