# PREREG - R-M2 resize-kernel diagnostic (locked 2026-09-27 ~01:00 IST, BEFORE any diagnostic fold)

Ordered by ChatGPT judge round 6 (isef_judge/malaria_round6.md): highest
information-per-hour experiment before R-M2. Question: is the R-M1 matched-protocol
failure partly a preprocessing (resize-kernel) confounder?

## Protocol (identical to R-M1 except the resize kernel)
- Data: same 27,558 NIH images, same patient-cluster folds as PREREG_RM1
  (rng 2026 assignment, verified ids/labels align across all dataset variants).
- Model: GlobalCNNClassifier from scratch, Adam lr=1e-3, batch 64, 8 epochs,
  patience 3, seed = fold + 42; early-stop val = 10% of train clusters (seed fold+7).
- Variants (only the raw->48x48 resize kernel differs):
  V0 pil_bilinear = existing malaria48 (R-M1 results stand, not re-run)
  V1 cv2_area     = malaria48_area
  V2 cv2_linear   = malaria48_linear
  V3 pil_lanczos  = malaria48_lanczos
- Metric: held-out fold accuracy @0.5 (same as R-M1), AUC context.

## Decision rule (locked)
- Let A(v) = 5-fold mean accuracy of variant v.
- If max(A) - A(V0) > 0.01: preprocessing is a confounder; R-M1's comparison is
  re-interpreted accordingly and PREREG_R-M2 adopts the best kernel, reported honestly.
- If max(A) - A(V0) <= 0.01: preprocessing is NOT the bottleneck; R-M2 proceeds on the
  stronger-backbone pivot (judge round 6 choice) with the V0 kernel.
- Either outcome is reported in the paper's R-M2 chapter; a null is a valid result.

## Cost note
Same compute shape as R-M1 (~15 min/fold on this box) x 3 new variants x 5 folds,
ground in checkpointed chunks alongside DAVIS/L2C lanes.
