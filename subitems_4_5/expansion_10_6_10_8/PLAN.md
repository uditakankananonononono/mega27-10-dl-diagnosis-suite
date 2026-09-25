# Item 10 expansion (user directive 21:39-21:41 IST, via parent)
Three NEW separate projects, each full gate set (40 external tools / 120
accession-level datasets / 50pp+ paper / benchmark win). Additive to
10.4 malaria + 10.5 pneumonia, which continue unchanged.

## 10.6 Cancer (proposed; numbering to be confirmed by 10a)
- Dataset lead: BreakHis (breast histopathology, 7,909 images, 8 classes,
  patient-level splits published) or PatchCamelyon (327k histopathology
  patches, benchmarked). Both public, identifier-backed.
- Benchmark to beat: published BreakHis image-level accuracies (~83-90%
  by magnification) or PCam baseline (~0.90 AUC) - source-verify first.
- Census angle: label noise in histopathology is documented territory;
  confident-learning census + per-image flags as per 10.4/10.5 recipe.

## 10.7 Neurological (proposed)
- Dataset lead: OASIS-1/2 (structural MRI, dementia labels, 400+ subjects)
  or Kaggle Brain Tumor MRI (7,023 images, 4 classes, widely benchmarked).
- Benchmark: published per-dataset baselines - source-verify first.
- Census angle: same recipe; 3D data may need slice-level adaptation.

## 10.8 Genetic (proposed)
- Dataset lead: ClinVar variant-effect records (identifier-backed, 1M+
  accessions; pathogenic/benign labels) - tabular/sequence, not imaging:
  CNN over sequence windows + GNN over variant graphs fits the lane's
  architecture contract.
- Benchmark: published variant-effect predictors with open evaluation
  (source-verify current SOTA claims first).

## Status at opening (21:41 IST): 0/40 tools, 0/120 datasets, 0pp papers.
Honest capacity note: malaria+pneumonia close first (tonight); expansion
projects start immediately after, not in parallel on this sandbox.
