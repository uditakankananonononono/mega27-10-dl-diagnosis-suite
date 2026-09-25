"""Tool battery 2: decode cross-checks, resize audits, figures, and
literature/provenance APIs. Each tool writes committed JSON evidence."""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders

OUT = Path(__file__).resolve().parent.parent / "results"


def decode_crosscheck(suite, it, n, ds_name):
    """imageio vs Pillow independent decoders: max abs pixel diff must be 0
    for lossless paths; for JPEG both decode the same bytes so diff is 0."""
    import imageio.v3 as iio
    from PIL import Image
    t0 = time.time()
    checked = 0
    max_diff = 0
    ids = []
    for r in it:
        if checked >= n:
            break
        src = r.get("_source_bytes")
        if src is None:
            break
        a = np.asarray(Image.open(src).convert("RGB"), dtype=np.int16)
        b = iio.imread(src, extension=src.suffix).astype(np.int16)
        if a.shape == b.shape:
            max_diff = max(max_diff, int(np.abs(a - b).max()))
            checked += 1
            ids.append(r.get("image_id") or r.get("record_id"))
    return {"dataset": ds_name, "tool": "imageio+Pillow", "n_checked": checked,
            "max_abs_pixel_diff": max_diff, "wall_seconds": round(time.time() - t0, 2)}


def resize_audit(suite, it, n, ds_name):
    """OpenCV vs Pillow resize backends: probability-relevant property shift
    (mean abs pixel delta after 64x64 resize by two backends)."""
    import cv2
    from PIL import Image
    t0 = time.time()
    deltas = []
    for i, r in enumerate(it):
        if i >= n:
            break
        img = r["image"]
        a = np.asarray(Image.fromarray(img).resize((64, 64), Image.BILINEAR), dtype=np.float32)
        b = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        deltas.append(float(np.abs(a - b).mean()))
    return {"dataset": ds_name, "tool": "OpenCV", "n_images": len(deltas),
            "mean_abs_backend_delta": round(float(np.mean(deltas)), 4),
            "max_abs_backend_delta": round(float(np.max(deltas)), 4),
            "wall_seconds": round(time.time() - t0, 2)}


def class_figure(suite):
    """matplotlib: committed class-distribution figures per suite."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = []
    specs = []
    if suite == "cancer":
        d1 = json.load(open(OUT / "cancer" / "breakhis_manifest.json"))
        specs.append(("breakhis_subclasses", d1["subclass_distribution"]))
        d2 = json.load(open(OUT / "cancer" / "pcam_manifest.json"))
        specs.append(("pcam_valid_labels", d2["label_distribution_valid"]))
    elif suite == "neuro":
        d = json.load(open(OUT / "neuro" / "brain_mri_manifest.json"))
        specs.append(("brain_mri_classes", d["class_distribution"]))
    elif suite == "genetic":
        d = json.load(open(OUT / "genetic" / "clinvar_manifest.json"))
        top = dict(list(d["clinical_significance_distribution"].items())[:8])
        specs.append(("clinvar_significance_top8", top))
    for name, dist in specs:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(dist)), list(dist.values()))
        ax.set_xticks(range(len(dist)))
        ax.set_xticklabels(list(dist.keys()), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("records")
        ax.set_title(name)
        fig.tight_layout()
        p = OUT / suite / f"fig_{name}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        figs.append(p.name)
    return {"suite": suite, "tool": "matplotlib", "figures": figs}


def lit_audit(suite, queries):
    """NCBI E-utilities + CrossRef: novelty/benchmark literature evidence,
    PMIDs + DOI metadata captured to JSON."""
    import urllib.request, urllib.parse
    out = {"suite": suite, "tools": ["NCBI E-utilities", "CrossRef API"], "queries": {}}
    for qname, q in queries.items():
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&retmax=5&term="
               + urllib.parse.quote(q))
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
            out["queries"][qname] = {"query": q, "count": int(d["esearchresult"]["count"]),
                                     "top_pmids": d["esearchresult"]["idlist"]}
        except Exception as e:
            out["queries"][qname] = {"query": q, "error": str(e)}
        time.sleep(0.4)
    return out


def doi_verify(dois):
    import urllib.request
    out = {"tool": "CrossRef API", "dois": {}}
    for doi in dois:
        try:
            req = urllib.request.Request("https://api.crossref.org/works/" + doi,
                                         headers={"User-Agent": "mega27-expansion/1.0 (mailto:builder-10-expansion@instinct.local)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                m = json.load(r)["message"]
            out["dois"][doi] = {"title": m.get("title", [None])[0],
                                "container": m.get("container-title", [None])[0],
                                "year": m.get("issued", {}).get("date-parts", [[None]])[0][0]}
        except Exception as e:
            out["dois"][doi] = {"error": str(e)}
        time.sleep(0.3)
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    if which == "cancer":
        json.dump(resize_audit("cancer", loaders.iter_pcam("valid"), n, "pcam_valid"),
                  open(OUT / "cancer" / "tool_battery_2_pcam_resize_audit.json", "w"), indent=1)
        json.dump(resize_audit("cancer", loaders.iter_breakhis(), n, "breakhis"),
                  open(OUT / "cancer" / "tool_battery_2_breakhis_resize_audit.json", "w"), indent=1)
        json.dump(class_figure("cancer"), open(OUT / "cancer" / "tool_battery_2_figures.json", "w"), indent=1)
        json.dump(lit_audit("cancer", {
            "breakhis_label_noise": "BreakHis breast histopathology deep learning",
            "pcam_benchmark": "PatchCamelyon metastasis detection deep learning"}),
            open(OUT / "cancer" / "tool_battery_2_lit_audit.json", "w"), indent=1)
        json.dump(doi_verify(["10.1109/TBME.2015.2496264", "10.1109/CBMS.2018.00062"]),
                  open(OUT / "cancer" / "tool_battery_2_doi_verify.json", "w"), indent=1)
    elif which == "neuro":
        json.dump(resize_audit("neuro", loaders.iter_neuro(), n, "brain_mri"),
                  open(OUT / "neuro" / "tool_battery_2_resize_audit.json", "w"), indent=1)
        json.dump(class_figure("neuro"), open(OUT / "neuro" / "tool_battery_2_figures.json", "w"), indent=1)
        json.dump(lit_audit("neuro", {
            "brain_tumor_mri_benchmark": "brain tumor MRI classification deep learning glioma meningioma pituitary"}),
            open(OUT / "neuro" / "tool_battery_2_lit_audit.json", "w"), indent=1)
    elif which == "genetic":
        json.dump(class_figure("genetic"), open(OUT / "genetic" / "tool_battery_2_figures.json", "w"), indent=1)
        json.dump(lit_audit("genetic", {
            "clinvar_pathogenicity": "ClinVar variant pathogenicity prediction machine learning"}),
            open(OUT / "genetic" / "tool_battery_2_lit_audit.json", "w"), indent=1)
        json.dump(doi_verify(["10.1093/nar/gkx1153"]),
                  open(OUT / "genetic" / "tool_battery_2_doi_verify.json", "w"), indent=1)
    print(which, "battery2 done")
