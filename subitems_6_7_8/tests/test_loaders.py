"""Hermetic tests for subitems_6_7_8 loaders. No network, no real data:
every fixture is synthesized in tmp_path."""
import csv, gzip, io, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import loaders  # noqa: E402


def _jpg(path, shape=(8, 8, 3)):
    from PIL import Image
    Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(path, format="JPEG")


def test_breakhis_loader(tmp_path):
    root = tmp_path / "breakhis" / "breakhis" / "histology_slides" / "breast"
    p1 = root / "benign" / "SOB" / "adenosis" / "SOB_B_A_14-22549AB" / "40X"
    p2 = root / "malignant" / "SOB" / "ductal_carcinoma" / "SOB_M_DC_14-10990" / "100X"
    p1.mkdir(parents=True); p2.mkdir(parents=True)
    _jpg(p1 / "SOB_B_A-14-22549AB-40-001.jpg")
    _jpg(p2 / "SOB_M_DC-14-10990-100-001.jpg")
    recs = list(loaders.iter_breakhis(root))
    assert len(recs) == 2
    by_label = {r["label"]: r for r in recs}
    assert by_label[0]["patient_id"] == "SOB_B_A_14-22549AB"
    assert by_label[0]["subclass_code"] == "A"
    assert by_label[1]["magnification"] == "100X"
    assert by_label[1]["subclass_code"] == "DC"
    assert all(r["image"].shape == (8, 8, 3) for r in recs)


def test_pcam_loader(tmp_path):
    import h5py
    x = np.zeros((2, 96, 96, 3), dtype=np.uint8)
    y = np.array([[[[0]]], [[[1]]]], dtype=np.uint8)
    with h5py.File(tmp_path / "camelyonpatch_level_2_split_valid_x.h5", "w") as f:
        f.create_dataset("x", data=x)
    with h5py.File(tmp_path / "camelyonpatch_level_2_split_valid_y.h5", "w") as f:
        f.create_dataset("y", data=y)
    with open(tmp_path / "camelyonpatch_level_2_split_valid_meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "coord_y", "coord_x", "tumor_patch", "center_tumor_patch", "wsi"])
        w.writerow([0, 100, 200, "False", "False", "wsi_a"])
        w.writerow([1, 300, 400, "True", "True", "wsi_b"])
    recs = list(loaders.iter_pcam("valid", tmp_path))
    assert len(recs) == 2
    assert recs[0]["patch_id"] == "pcam_valid_000000"
    assert recs[0]["label"] == 0 and recs[1]["label"] == 1
    assert recs[1]["wsi"] == "wsi_b" and recs[1]["coord"] == (300, 400)
    assert recs[0]["image"].shape == (96, 96, 3)


def test_neuro_loader(tmp_path):
    import pyarrow as pa, pyarrow.parquet as pq
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.zeros((6, 6, 3), dtype=np.uint8)).save(buf, format="PNG")
    t = pa.table({"image": [{"bytes": buf.getvalue(), "path": "a.png"},
                            {"bytes": buf.getvalue(), "path": "b.png"}],
                  "label": [0, 3]})
    pq.write_table(t, tmp_path / "brain_tumor_mri_train.parquet")
    recs = list(loaders.iter_neuro(tmp_path))
    assert len(recs) == 2
    assert recs[0]["record_id"] == "btmri_00000" and recs[0]["class_name"] == "glioma"
    assert recs[1]["class_name"] == "pituitary"
    assert recs[0]["image"].shape == (6, 6, 3)


def test_clinvar_subset_loader(tmp_path):
    gz = tmp_path / "sub.tsv.gz"
    cols = ["VariationID", "AlleleID", "Type", "ClinicalSignificance", "ClinSigSimple",
            "GeneSymbol", "GeneID", "RCVaccession", "OriginSimple", "Assembly",
            "Chromosome", "Start", "Stop", "ReferenceAlleleVCF", "AlternateAlleleVCF",
            "ReviewStatus", "PhenotypeList", "RS# (dbSNP)", "Name"]
    rows = [["11", "22", "single nucleotide variant", "Pathogenic", "1", "BRCA1", "672",
             "RCV0001", "germline", "GRCh38", "17", "100", "100", "A", "G",
             "criteria provided, single submitter", "breast cancer", "rs1", "NM_1:c.1A>G"],
            ["12", "23", "single nucleotide variant", "Benign", "0", "TP53", "7157",
             "RCV0002", "germline", "GRCh38", "17", "200", "200", "C", "T",
             "criteria provided, single submitter", "none", "rs2", "NM_2:c.2C>T"]]
    with gzip.open(gz, "wt") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    recs = list(loaders.iter_clinvar_subset(gz))
    assert len(recs) == 2
    assert recs[0]["label"] == 1 and recs[0]["variation_id"] == "11"
    assert recs[1]["label"] == 0 and recs[1]["gene"] == "TP53"
    assert recs[0]["ref"] == "A" and recs[0]["alt"] == "G"


def test_manifests_schema():
    res = Path(__file__).resolve().parent.parent / "results"
    man = list(res.rglob("*_manifest.json"))
    if not man:  # manifests are committed artifacts; skip before first build
        pytest.skip("no manifests built yet")
    import json
    for m in man:
        d = json.load(open(m))
        n = d.get("n_records", d.get("n_rows"))  # image manifests: n_records; clinvar tabular: n_rows
        assert n and n > 0, m
        assert "source" in d and "suite" in d, m

