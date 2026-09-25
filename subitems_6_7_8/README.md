# MEGA27 Item 10.6-10.8: Expansion Diagnosis Suites (Cancer, Neuro, Genetic)

Owner: task lane builder-10-expansion (branch `builder-10-expansion` only).
Additive to 10.4 malaria + 10.5 pneumonia (`subitems_4_5/`), which continue
unchanged. Follows the subitems_4_5 per-disease pattern: accession-level
manifests, hash-verified sources, committed JSON evidence for every tool.

## Datasets (all real, hash-verified, accession-level)
- **10.6 Cancer**: BreakHis v1 breast histopathology (7,909 images, 82
  patients, 8 subclasses, 4 magnifications; hirundo-io HF mirror, sha256
  matches HF LFS oid; CAVEAT: 224x224 JPG resize, not 700x460 PNG) +
  PatchCamelyon valid+test splits (65,536 patches from unique WSIs, official
  Zenodo mirror).
- **10.7 Neuro**: Brain Tumor MRI (Masoud Nickparvar), 7,023 images, 4
  classes (glioma/meningioma/notumor/pituitary); HF mirror
  Hemg/Brain-Tumor-MRI-Dataset, 512x512.
- **10.8 Genetic**: NCBI ClinVar variant_summary (9,224,097 rows; unique
  VariationIDs + RCV accessions in manifest) + curated germline SNV
  classification subset (pathogenic vs benign, >=1-star review).

## Reproduce
```
pip install torch torchvision scikit-learn pytest h5py pyarrow pillow --index-url https://download.pytorch.org/whl/cpu
python -m pytest tests/ -q              # hermetic, no network
python src/build_manifests.py all       # rebuild manifests from data/
python src/verify_loads.py breakhis 200 # real-data decode smoke (also pcam|neuro|clinvar)
```
Results land in `results/<suite>/*.json` (committed). Raw data stays in
gitignored `data/`.
