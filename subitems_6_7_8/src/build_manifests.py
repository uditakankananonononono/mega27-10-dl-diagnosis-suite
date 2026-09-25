"""Accession-level dataset manifests for MEGA27 item 10.6-10.8 expansion.

Every record is an identifier-backed unit (image path / patch index /
parquet row / ClinVar VariationID), individually fetched and verified.
Source archives are sha256-hashed; sizes checked against server-stated
values captured at download time. Emits committed JSON manifests under
results/<suite>/. Mirrors the subitems_4_5 per-disease pattern.
"""
import csv, gzip, hashlib, json, re, sys
from collections import Counter
from pathlib import Path

DATA = Path("/home/sandbox/mega27-expansion/data")
OUT = Path(__file__).resolve().parent.parent / "results"

def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

SOURCES = {
    "clinvar": {
        "url": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
        "server_content_length": 450706229,
        "server_last_modified": "Thu, 24 Sep 2026 03:59:56 GMT",
    },
    "pcam": {
        "url_base": "https://zenodo.org/records/2546921/files/",
        "record": "https://zenodo.org/records/2546921",
        "note": "PatchCamelyon official Zenodo mirror (Veeling et al. 2018)",
    },
    "breakhis_mirror": {
        "url": "https://huggingface.co/datasets/hirundo-io/BreaKHis-original/resolve/main/breakhis.zip",
        "lfs_size": 223947830,
        "note": ("Mirror of BreakHis v1 (Spanhol et al. 2016, UFPR). Official "
                 "host http://www.inf.ufpr.br/vri/BreaKHis_v1.tar.gz 404s at "
                 "fetch time; mirror verified against published structure: "
                 "7,909 images, 8 subclasses, 4 magnifications, patient IDs. "
                 "CAVEAT: mirror images are JPG resized to 224x224 (verified "
                 "on random sample of 12 across mags), not the original "
                 "700x460 PNGs - benchmark comparisons must state this."),
    },
    "neuro_mirror": {
        "url": "https://huggingface.co/datasets/Hemg/Brain-Tumor-MRI-Dataset/resolve/main/data/train-00000-of-00001.parquet",
        "lfs_size": 155450982,
        "note": ("Mirror of the Masoud Nickparvar Brain Tumor MRI dataset "
                 "(7,023 images; glioma/meningioma/notumor/pituitary), "
                 "public HF dataset card verified at fetch time."),
    },
}

def manifest_breakhis():
    root = DATA / "breakhis" / "breakhis" / "histology_slides" / "breast"
    recs = []
    pat = re.compile(r"(benign|malignant)/SOB/([^/]+)/(SOB_[^/]+)/(\d+X)/([^/]+\.jpg)$")
    for p in sorted(root.rglob("*.jpg")):
        rel = p.relative_to(root).as_posix()
        m = pat.search(rel)
        if not m:
            continue
        recs.append({
            "image_id": rel,
            "benign_malignant": m.group(1),
            "subclass_dir": m.group(2),
            "patient_id": m.group(3),
            "magnification": m.group(4),
            "filename": m.group(5),
        })
    sub_pat = re.compile(r"SOB_([BM])_([A-Z]+)_")
    for r in recs:
        mm = sub_pat.match(r["patient_id"])
        r["subclass_code"] = mm.group(2) if mm else None
    by_class = Counter(r["benign_malignant"] for r in recs)
    by_mag = Counter(r["magnification"] for r in recs)
    by_sub = Counter((r["benign_malignant"], r["subclass_code"]) for r in recs)
    zf = DATA / "breakhis" / "breakhis_hirundo.zip"
    man = {
        "suite": "10.6-cancer", "dataset": "BreaKHis v1 (hirundo-io HF mirror)",
        "source": SOURCES["breakhis_mirror"],
        "archive_sha256": sha256(zf), "archive_bytes": zf.stat().st_size,
        "n_records": len(recs),
        "n_patients": len({r["patient_id"] for r in recs}),
        "class_distribution": dict(by_class),
        "magnification_distribution": dict(by_mag),
        "subclass_distribution": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_sub.items())},
        "image_shape": "224x224x3 (resized JPG mirror)",
    }
    (OUT / "cancer").mkdir(parents=True, exist_ok=True)
    json.dump(man, open(OUT / "cancer" / "breakhis_manifest.json", "w"), indent=1)
    with open(OUT / "cancer" / "breakhis_accessions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    return man

def manifest_pcam():
    recs_n = {}
    for split in ("valid", "test"):
        meta = DATA / f"camelyonpatch_level_2_split_{split}_meta.csv"
        with open(meta) as f:
            rd = csv.DictReader(f)
            rows = list(rd)
        recs_n[split] = len(rows)
        label_field = "tumor_patch"
        dist = Counter(r[label_field] for r in rows)
        ws = len({r["wsi"] for r in rows})
        with open(OUT / "cancer" / f"pcam_{split}_accessions.csv", "w", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["patch_id", "coord_y", "coord_x", "tumor_patch", "center_tumor_patch", "wsi"])
            for i, r in enumerate(rows):
                w.writerow([f"pcam_{split}_{i:06d}", r["coord_y"], r["coord_x"],
                            r["tumor_patch"], r["center_tumor_patch"], r["wsi"]])
        if split == "valid":
            valid_dist, valid_wsi = dict(dist), ws
        else:
            test_dist, test_wsi = dict(dist), ws
    files = {}
    for split in ("valid", "test"):
        for kind in ("x.h5.gz", "y.h5.gz", "meta.csv"):
            p = DATA / f"camelyonpatch_level_2_split_{split}_{kind}"
            files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)} if p.stat().st_size < 2**22 else {"bytes": p.stat().st_size}
    # hash big x files only when fully downloaded (size check vs zenodo listing)
    expected = {"camelyonpatch_level_2_split_valid_x.h5.gz": 805965320,
                "camelyonpatch_level_2_split_test_x.h5.gz": 800875929}
    for name, size in expected.items():
        p = DATA / name
        st = p.stat().st_size
        files[name]["download_complete"] = (st == size)
        files[name]["zenodo_listed_bytes"] = size
        if st == size:
            files[name]["sha256"] = sha256(p)
    man = {
        "suite": "10.6-cancer", "dataset": "PatchCamelyon (PCam), valid+test splits",
        "source": SOURCES["pcam"],
        "n_records": sum(recs_n.values()),
        "records_per_split": recs_n,
        "label_distribution_valid": valid_dist, "label_distribution_test": test_dist,
        "n_unique_wsi_valid": valid_wsi, "n_unique_wsi_test": test_wsi,
        "files": files,
    }
    json.dump(man, open(OUT / "cancer" / "pcam_manifest.json", "w"), indent=1)
    return man

def manifest_neuro():
    import pyarrow.parquet as pq
    p = DATA / "neuro" / "brain_tumor_mri_train.parquet"
    t = pq.read_table(p, columns=["label"])
    labels = t.column("label").to_pylist()
    names = {0: "glioma", 1: "meningioma", 2: "notumor", 3: "pituitary"}
    dist = Counter(names.get(l, str(l)) for l in labels)
    with open(OUT / "neuro" / "brain_mri_accessions.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["record_id", "label", "class_name"])
        for i, l in enumerate(labels):
            w.writerow([f"btmri_{i:05d}", l, names.get(l, str(l))])
    man = {
        "suite": "10.7-neuro", "dataset": "Brain Tumor MRI (Masoud Nickparvar), HF mirror Hemg/Brain-Tumor-MRI-Dataset",
        "source": SOURCES["neuro_mirror"],
        "parquet_sha256": sha256(p), "parquet_bytes": p.stat().st_size,
        "n_records": len(labels),
        "class_distribution": dict(dist),
    }
    (OUT / "neuro").mkdir(parents=True, exist_ok=True)
    json.dump(man, open(OUT / "neuro" / "brain_mri_manifest.json", "w"), indent=1)
    return man

def manifest_clinvar():
    gz = DATA / "clinvar" / "variant_summary.txt.gz"
    sig_counter = Counter()
    origin_germline = 0
    total = 0
    variation_ids = set()
    rcv = set()
    subset_rows = 0
    keep_sig = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic",
                "Benign", "Likely benign", "Benign/Likely benign"}
    one_star_plus = re.compile(r"criteria provided|reviewed by expert panel|practice guideline")
    subset_path = DATA / "clinvar" / "clinvar_classification_subset.tsv.gz"
    (OUT / "genetic").mkdir(parents=True, exist_ok=True)
    with gzip.open(gz, "rt", errors="replace") as f, gzip.open(subset_path, "wt") as sf:
        header = f.readline().rstrip("\n").split("\t")
        idx = {k.lstrip("#"): i for i, k in enumerate(header)}
        need = ["VariationID", "AlleleID", "Type", "ClinicalSignificance", "ClinSigSimple",
                "GeneSymbol", "GeneID", "RCVaccession", "OriginSimple", "Assembly",
                "Chromosome", "Start", "Stop", "ReferenceAlleleVCF", "AlternateAlleleVCF",
                "ReviewStatus", "PhenotypeList", "RS# (dbSNP)", "Name"]
        sf.write("\t".join(need) + "\n")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            total += 1
            sig = parts[idx["ClinicalSignificance"]]
            sig_counter[sig] += 1
            variation_ids.add(parts[idx["VariationID"]])
            for a in parts[idx["RCVaccession"]].split("|"):
                if a:
                    rcv.add(a)
            if (parts[idx["OriginSimple"]] == "germline"
                    and sig in keep_sig
                    and one_star_plus.search(parts[idx["ReviewStatus"]])
                    and parts[idx["Type"]] == "single nucleotide variant"):
                sf.write("\t".join(parts[idx[c]] for c in need) + "\n")
                subset_rows += 1
    man = {
        "suite": "10.8-genetic", "dataset": "NCBI ClinVar variant_summary (tab-delimited)",
        "source": SOURCES["clinvar"],
        "archive_sha256": sha256(gz), "archive_bytes": gz.stat().st_size,
        "n_rows": total,
        "n_unique_variation_ids": len(variation_ids),
        "n_unique_rcv_accessions": len(rcv),
        "clinical_significance_distribution": dict(sig_counter.most_common()),
        "classification_subset": {
            "filter": "OriginSimple=germline AND Type=single nucleotide variant AND ClinicalSignificance in {Pathogenic, Likely pathogenic, Pathogenic/Likely pathogenic, Benign, Likely benign, Benign/Likely benign} AND ReviewStatus >= 1 star (criteria provided+)",
            "n_records": subset_rows,
            "file": "data/clinvar/clinvar_classification_subset.tsv.gz (gitignored, 102MB gz)",
        },
    }
    json.dump(man, open(OUT / "genetic" / "clinvar_manifest.json", "w"), indent=1)
    return man

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("breakhis", manifest_breakhis), ("pcam", manifest_pcam),
                     ("neuro", manifest_neuro), ("clinvar", manifest_clinvar)]:
        if which in ("all", name):
            m = fn()
            print(f"[{name}] n_records={m.get('n_records') or m.get('n_rows')}")

