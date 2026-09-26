# PREREG R-M1: matched-protocol malaria beat rung (locked 2026-09-26 17:47 IST, before any fold)
Target: Rajaraman et al. 2018 (PeerJ 6:e4568) best published accuracy
0.957 +/- 0.007 under 5-fold PATIENT-level cross-validation on the full
27,558 NIH cell images (Table 2, fetched live 2026-09-26; beat
verification in results/rajaraman_beat_verification.json).

## Protocol (matched to comparator)
- Clusters: patient/smear code = filename segment before "_IMG"
  (verified: 201 clusters over 27,558 images; matches the census's
  clustering unit).
- Folds: 5 folds at cluster level. Assignment: clusters shuffled
  (np.random.default_rng(2026).permutation), dealt round-robin into 5
  folds; no image of a cluster crosses folds. (Rajaraman's exact fold
  assignments are not published; matched = same protocol family + n=5.)
- Model per fold: GlobalCNNClassifier (committed architecture), from
  scratch, Adam lr=1e-3, batch 64, epochs 8, patience 3, seed = fold
  index + 42. Early-stopping val = 10% of the fold's TRAIN clusters
  (rng seed fold+7, stratified by class at image level).
- Metric: accuracy on the held-out fold's images (threshold 0.5), AUC.
- GATE (clean beat): 5-fold mean accuracy > 0.957 AND the 95% t-interval
  of the 5 fold accuracies (df=4) excludes 0.957. Also reported: mean
  AUC vs their 0.990 (context, not the gate).
- Compute honesty: all folds run in THIS environment (Pillow 12.3.0,
  torch 2.14 cpu); results env-consistent per
  results/malaria/provenance_verification.json. If gate fails: honest
  negative -> R-M2 (ImageNet-transfer arm) preregistered separately,
  with RULE-6 ChatGPT redirection input.
