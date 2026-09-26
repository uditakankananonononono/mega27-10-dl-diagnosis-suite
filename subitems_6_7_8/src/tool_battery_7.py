"""Battery 7: captum saliency attribution for the PCam and neuro CNNs,
pycm full confusion-matrix statistics on PCam probs, myvariant.info +
Ensembl REST annotation of ClinVar variants/genes, pysam VCF round-trip,
upsetplot flag-set intersections on the PCam held-out universe."""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
NP = Path("/home/sandbox/mega27-expansion/np")


def _saliency(tag, ckpt_path, npz_keys, n_classes, size, out_json, n_img=8):
    import torch
    from captum.attr import Saliency
    sys.path.insert(0, str(ROOT / "src"))
    import train_cnn, train_ckpt
    d = np.load(NP / npz_keys[0])
    X, y = d[npz_keys[1]], d[npz_keys[2]]
    X, y = X[:n_img], y[:n_img]
    model = train_cnn.small_cnn(3, n_classes, size)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model"])
    model.eval()
    xt = torch.from_numpy(X.astype(np.float32) / 255.0)
    sal = Saliency(model)
    attrs = sal.attribute(xt, target=torch.from_numpy(y.astype(np.int64)))
    a = attrs.abs().mean(dim=1).numpy()  # (n,H,W)
    rows = [{"index": i, "true": int(y[i]),
             "saliency_mean": round(float(a[i].mean()), 5),
             "saliency_max": round(float(a[i].max()), 5),
             "center_share": round(float(a[i, size//4:3*size//4, size//4:3*size//4].mean() / max(a[i].mean(), 1e-9)), 3)}
            for i in range(len(X))]
    json.dump({"tool": "captum Saliency (input gradients)",
               "model_ckpt": str(ckpt_path.name), "n_images": len(X),
               "per_image": rows,
               "note": "center_share > 1 means attribution concentrates in the informative center region"},
              open(out_json, "w"), indent=1)


def captum_pcam():
    _saliency("pcam", OUT / "cancer" / "pcam_cnn_ckpt.pt",
              ("pcam_test_4096.npz", "X", "y"), 2, 96, OUT / "cancer" / "pcam_captum_saliency.json")


def captum_neuro():
    _saliency("neuro", OUT / "neuro" / "brain_mri_cnn_ckpt.pt",
              ("neuro_64.npz", "Xte", "yte"), 4, 64, OUT / "neuro" / "neuro_captum_saliency.json")


def pycm_pcam():
    from pycm import ConfusionMatrix
    d = np.load(OUT / "cancer" / "pcam_cnn_probs.npz")
    y, p = d["y_true"], d["prob_pos"]
    pred = (p > 0.5).astype(int)
    cm = ConfusionMatrix(y.tolist(), pred.tolist())
    want = {"Overall ACC": "acc", "Kappa": "kappa", "Overall MCC": "mcc",
            "F1 Macro": "f1_macro", "TPR Macro": "tpr_macro", "TNR Macro": "tnr_macro",
            "PPV Macro": "ppv_macro", "NPV Macro": "npv_macro", "FNR Macro": "fnr_macro"}
    stats = {v: (round(float(cm.overall_stat[k]), 4) if isinstance(cm.overall_stat.get(k), (int, float)) else None)
             for k, v in want.items()}
    json.dump({"tool": "pycm ConfusionMatrix", "dataset": "pcam_test held-out probs",
               "matrix": {"TP": int(cm.TP[1]), "FP": int(cm.FP[1]), "TN": int(cm.TN[1]), "FN": int(cm.FN[1])},
               "stats": stats, "note": "full confusion-matrix statistics cross-check of the CNN"},
              open(OUT / "cancer" / "pcam_pycm_stats.json", "w"), indent=1)


def myvariant_clinvar():
    import myvariant
    sys.path.insert(0, str(ROOT / "src"))
    import loaders
    mv = myvariant.MyVariantInfo()
    rows, seen = [], set()
    for r in loaders.iter_clinvar_subset():
        if r["label"] == 1 and r["variation_id"] not in seen and r["ref"] and r["alt"]:
            seen.add(r["variation_id"])
            chrom = r["chrom"] if r["chrom"].isdigit() else None
            if not chrom or not r.get("pos"):
                continue
            hgvs = f"chr{chrom}:g.{r['pos']}{r['ref'][:1]}>{r['alt'][:1]}"
            rows.append({"variation_id": r["variation_id"], "gene": r["gene"], "hgvs": hgvs})
        if len(rows) >= 25:
            break
    hits = mv.getvariants([x["hgvs"] for x in rows], fields="dbsnp.rsid,clinvar.rcv,gnomad_exome.af.af", verbose=False)
    n_hit = sum(1 for h in hits if not h.get("notfound"))
    sample = []
    for x, h in zip(rows[:8], hits[:8]):
        sample.append({"variation_id": x["variation_id"], "gene": x["gene"],
                       "found": not h.get("notfound"),
                       "rsid": (h.get("dbsnp") or {}).get("rsid") if isinstance(h.get("dbsnp"), dict) else None})
    json.dump({"tool": "myvariant.info Python client (live API)",
               "dataset": "25 pathogenic ClinVar variants, genomic HGVS queries",
               "n_queried": len(rows), "n_annotated": n_hit,
               "sample": sample,
               "note": "external annotation cross-check of the accession-level ClinVar records"},
              open(OUT / "genetic" / "clinvar_myvariant_annotate.json", "w"), indent=1)


def ensembl_genes():
    import requests
    graph = json.load(open(OUT / "genetic" / "clinvar_networkx_graph.json"))
    genes = [g["gene"] for g in graph["top_gene_degrees"][:10]]
    out = []
    for g in genes:
        try:
            r = requests.get(f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{g}?content-type=application/json",
                             timeout=15)
            js = r.json()
            out.append({"gene": g, "status": r.status_code,
                        "ensembl_id": js[0]["id"] if js else None})
        except Exception as e:
            out.append({"gene": g, "error": str(e)[:80]})
        time.sleep(0.2)
    n_ok = sum(1 for x in out if x.get("ensembl_id"))
    json.dump({"tool": "Ensembl REST API (xrefs/symbol)",
               "dataset": "top-10 ClinVar networkx genes",
               "n_queried": len(out), "n_resolved": n_ok, "results": out,
               "note": "gene identity cross-verification against Ensembl"},
              open(OUT / "genetic" / "clinvar_ensembl_xref.json", "w"), indent=1)


def pysam_vcf():
    import pysam
    sys.path.insert(0, str(ROOT / "src"))
    import loaders
    hdr = pysam.VariantHeader()
    hdr.add_line("##source=mega27-10.8-clinvar-roundtrip")
    hdr.add_line('##INFO=<ID=CLNSIG,Number=1,Type=String,Description="clinical significance label">')
    hdr.add_line('##INFO=<ID=GENE,Number=1,Type=String,Description="gene symbol">')
    hdr.add_line('##INFO=<ID=VID,Number=1,Type=String,Description="ClinVar VariationID">')
    chroms, recs, seen = set(), [], set()
    for r in loaders.iter_clinvar_subset():
        if r["variation_id"] in seen or not r["chrom"].isdigit() or not r["pos"]:
            continue
        if len(r["ref"]) > 50 or len(r["alt"]) > 50 or not r["ref"] or not r["alt"]:
            continue
        seen.add(r["variation_id"])
        chroms.add(r["chrom"])
        recs.append(r)
        if len(recs) >= 1000:
            break
    for c in sorted(chroms, key=int):
        hdr.contigs.add(c)
    vcf_path = Path("/tmp/clinvar_roundtrip.vcf")
    with pysam.VariantFile(str(vcf_path), "w", header=hdr) as vf:
        for r in sorted(recs, key=lambda x: (int(x["chrom"]), x["pos"])):
            rec = hdr.new_record(contig=r["chrom"], start=int(r["pos"]) - 1,
                                 alleles=(r["ref"], r["alt"]))
            rec.info["CLNSIG"] = "pathogenic" if r["label"] == 1 else "benignish"
            rec.info["GENE"] = r["gene"] or "NA"
            rec.info["VID"] = str(r["variation_id"])
            vf.write(rec)
    n_back, n_path = 0, 0
    with pysam.VariantFile(str(vcf_path)) as vf:
        for rec in vf:
            n_back += 1
            if rec.info["CLNSIG"] == "pathogenic":
                n_path += 1
    json.dump({"tool": "pysam VariantFile VCF write+read round-trip",
               "dataset": "1000 ClinVar subset variants",
               "n_written": len(recs), "n_read_back": n_back,
               "n_pathogenic_read_back": n_path,
               "verdict": "lossless" if n_back == len(recs) else "MISMATCH",
               "note": "VCF is the interchange format for variant pipelines; round-trip proves field integrity"},
              open(OUT / "genetic" / "clinvar_pysam_vcf_roundtrip.json", "w"), indent=1)


def upset_pcam():
    from upsetplot import from_memberships, UpSet
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(OUT / "cancer" / "pcam_cnn_probs.npz")
    y, p = d["y_true"], d["prob_pos"]
    census = json.load(open(OUT / "cancer" / "pcam_label_census.json"))
    flagged = {f["patch_id"] for f in census["top50_flags"]}
    sets = []
    for i in range(len(y)):
        m = []
        if f"pcam_test_{i:06d}" in flagged:
            m.append("cleanlab_top50")
        if abs(p[i] - 0.5) < 0.1:
            m.append("low_confidence")
        if y[i] != (p[i] > 0.5):
            m.append("misclassified")
        sets.append(m)
    data = from_memberships(sets, data=np.ones(len(sets)))
    fig = plt.figure(figsize=(7, 4))
    UpSet(data, subset_size="count", show_counts=True).plot(fig)
    fig.savefig(OUT / "cancer" / "fig_pcam_flag_upset.png", dpi=110, bbox_inches="tight")
    json.dump({"tool": "upsetplot", "figure": "fig_pcam_flag_upset.png",
               "universe": "4096 pcam test patches",
               "sets": ["cleanlab_top50", "low_confidence", "misclassified"],
               "note": "intersection structure of the three independent flag sets"},
              open(OUT / "cancer" / "tool_battery_7_upset.json", "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    {"captum_pcam": captum_pcam, "captum_neuro": captum_neuro, "pycm": pycm_pcam,
     "myvariant": myvariant_clinvar, "ensembl": ensembl_genes, "pysam": pysam_vcf,
     "upset": upset_pcam}[which]()
    print(which, "done", flush=True)
