"""Accession-level loaders for the 10.6-10.8 expansion datasets.

Each loader yields identifier-backed records. No network at load time:
loaders read the verified local artifacts whose hashes are committed in
results/<suite>/*_manifest.json.
"""
import csv, gzip, io
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"


def iter_breakhis(root=None):
    """Yield BreakHis records: (image_id, patient_id, label, subclass_code,
    magnification, np.ndarray HWC uint8). label: 0 benign, 1 malignant."""
    from PIL import Image
    root = Path(root) if root else DATA / "breakhis" / "breakhis" / "histology_slides" / "breast"
    import re
    pat = re.compile(r"(benign|malignant)/SOB/([^/]+)/(SOB_[^/]+)/(\d+X)/([^/]+\.jpg)$")
    sub_pat = re.compile(r"SOB_([BM])_([A-Z]+)_")
    for p in sorted(root.rglob("*.jpg")):
        rel = p.relative_to(root).as_posix()
        m = pat.search(rel)
        if not m:
            continue
        mm = sub_pat.match(m.group(3))
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        yield {
            "image_id": rel, "patient_id": m.group(3),
            "label": 1 if m.group(1) == "malignant" else 0,
            "subclass_code": mm.group(2) if mm else None,
            "magnification": m.group(4), "image": img,
        }


def iter_pcam(split="valid", root=None):
    """Yield PCam records: (patch_id, label, wsi, coord, np.ndarray 96x96x3).
    Requires the split's decompressed x .h5 and y .h5 next to the meta csv."""
    import h5py
    root = Path(root) if root else DATA
    meta = root / f"camelyonpatch_level_2_split_{split}_meta.csv"
    x = h5py.File(root / f"camelyonpatch_level_2_split_{split}_x.h5", "r")["x"]
    y = h5py.File(root / f"camelyonpatch_level_2_split_{split}_y.h5", "r")["y"]
    with open(meta) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == x.shape[0] == y.shape[0], (len(rows), x.shape, y.shape)
    for i, r in enumerate(rows):
        yield {
            "patch_id": f"pcam_{split}_{i:06d}",
            "label": int(y[i].item()),
            "wsi": r["wsi"], "coord": (int(r["coord_y"]), int(r["coord_x"])),
            "image": x[i],
        }


def iter_neuro(root=None):
    """Yield Brain Tumor MRI records: (record_id, label, class_name,
    np.ndarray HWC uint8) from the HF parquet."""
    import pyarrow.parquet as pq
    from PIL import Image
    root = Path(root) if root else DATA / "neuro"
    t = pq.read_table(root / "brain_tumor_mri_train.parquet")
    names = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}
    images = t.column("image").to_pylist()
    labels = t.column("label").to_pylist()
    for i, (im, l) in enumerate(zip(images, labels)):
        arr = np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"), dtype=np.uint8)
        yield {"record_id": f"btmri_{i:05d}", "label": int(l),
               "class_name": names.get(int(l), str(l)), "image": arr}


def iter_clinvar_subset(path=None):
    """Yield curated ClinVar classification records (see clinvar_manifest.json
    for the exact filter). label: 1 pathogenic-side, 0 benign-side."""
    path = Path(path) if path else DATA / "clinvar" / "clinvar_classification_subset.tsv.gz"
    pos = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
    with gzip.open(path, "rt", errors="replace") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {k.lstrip("#"): i for i, k in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            sig = parts[idx["ClinicalSignificance"]]
            yield {
                "variation_id": parts[idx["VariationID"]],
                "allele_id": parts[idx["AlleleID"]],
                "rcv": parts[idx["RCVaccession"]],
                "gene": parts[idx["GeneSymbol"]],
                "label": 1 if sig in pos else 0,
                "clinical_significance": sig,
                "ref": parts[idx["ReferenceAlleleVCF"]],
                "alt": parts[idx["AlternateAlleleVCF"]],
                "chrom": parts[idx["Chromosome"]],
                "name": parts[idx["Name"]],
            }
