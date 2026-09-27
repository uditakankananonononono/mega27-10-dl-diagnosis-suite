# PREREG R-M2: stronger-backbone malaria beat rung (locked 2026-09-27 ~20:35 IST, BEFORE any R-M2 fold)

Target unchanged: Rajaraman et al. 2018 (PeerJ 6:e4568) best published accuracy
0.957 +/- 0.007 under 5-fold PATIENT-level cross-validation on the full
27,558 NIH cell images (results/rajaraman_beat_verification.json).

## Lineage (locked)
- R-M1 (matched protocol, GlobalCNNClassifier) FAILED its gate: mean 0.94771 < 0.957,
  95% t-CI [0.9226, 0.9728] (results/malaria/rm1/VERDICT.json). Honest negative.
- RULE-6 redirection (isef_judge/malaria_round6.md, ChatGPT consult): first run the
  resize-kernel diagnostic, then a stronger-backbone rung.
- PREREG_RESIZE_DIAG.md outcome (15/15 folds, results/malaria/resize_diag/):
  A(V0 bilinear)=0.94771, A(cv2_area)=0.9493, A(cv2_linear)=0.9412,
  A(pil_lanczos)=0.9510. max(A)-A(V0) = 0.0033 <= 0.01 -> NULL: preprocessing is
  NOT the bottleneck. Per the locked decision rule, R-M2 uses the V0 kernel and
  the stronger-backbone pivot. The diagnostic null is reported in the paper.

## Protocol (identical to R-M1 except the model class)
- Data: malaria48 (PIL bilinear 48x48, the V0 variant); same 27,558 images; same
  patient-cluster folds (rng 2026 assignment, identical to R-M1 and the diagnostic).
- Model per fold: ResNet-18 class backbone (torchvision resnet18, weights=None,
  num_classes=2), FROM SCRATCH - no pretrained weights, no external data (judge
  round-6 decision #2).
- Optimizer: Adam lr=1e-3, batch 64, 8 epochs, seed = fold + 42; deterministic
  per-epoch batch order (seeded generator 1000*fold+ep, resume-safe batch
  checkpoints identical to run_rm1.py). Early-stop val = 10% of train clusters
  (rng seed fold+7), used for logging exactly as R-M1 (R-M1 ran its full 8 epochs;
  R-M2 matches that implementation).
- Metric: held-out fold accuracy @0.5 (primary), AUC (context).

## GATE (clean beat), unchanged from R-M1
5-fold mean accuracy > 0.957 AND the 95% t-interval of the 5 fold accuracies
(df=4) excludes 0.957. Also reported: paired per-fold deltas vs R-M1 and a
bootstrap CI on the mean delta (context, not the gate).

## Falsification
If the gate fails: honest negative, published as such; next redirection consult
per RULE 6. AUC reframing remains rejected (round-6 decision #4).
