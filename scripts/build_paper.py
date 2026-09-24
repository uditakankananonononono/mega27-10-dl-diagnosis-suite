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
