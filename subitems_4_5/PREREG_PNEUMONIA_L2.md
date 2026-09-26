# PREREGISTRATION - Pneumonia L2: CXR-domain transfer ladder + census-cleaning ablation

Locked: 2026-09-26, BEFORE any L2 training or test evaluation. Trigger: ChatGPT judge round 3
(isef_judge/pneumonia_round3.md) redirection after the L1 honest negative (frozen ResNet18 +
logistic head: OOF train 0.9669, test 0.7821 @ frozen threshold 0.30 vs Kermany 0.928 target;
committed 33fa138). Judge's verdict: bottleneck is distribution shift, not capacity; do not add
architecture innovation. This prereg fixes the L2 protocol before outcomes exist.

## Frozen protocol (all models)
- Data: Kermany chest X-ray mirror (HF ahulikal/chest-xray-pneumonia-mirror, bit-exact verified,
  commit 9f203ac). OFFICIAL patient-level train/val/test split, unchanged. No patient crosses splits.
- Threshold: chosen on VALIDATION only, then frozen before the single test evaluation per model.
- Metrics: accuracy, ROC AUC, ECE, Brier. One test evaluation per model; no test-time tuning.
- Seeds: 3 fixed seeds for any trained head/fine-tune; report mean +/- sd.
- Falsification: if Model B does not beat L1 (0.7821) by >= 0.03 accuracy, the transfer premise
  fails for this pipeline; record honestly and redirect via the ChatGPT rule.

## Ladder
- Model A (reference, ALREADY RUN pre-prereg as L1): frozen ImageNet ResNet18 + logistic head.
  Test 0.7821. Documented as pre-existing; not re-run.
- Model B: frozen CXR-pretrained encoder + linear head. Encoder: TorchXRayVision
  densenet121-res224-all (free, downloadable weights; pediatric-adult domain closer than ImageNet).
- Model C: Model B encoder progressively fine-tuned (classifier only -> last block -> full, small lr).
- Model D: Model C recipe retrained with census-guided label treatment: (D1) consensus-core flags
  removed; (D2) original labels. Ablation, same everything.

## Preregistered hypotheses
- H1: Model B test accuracy >= 0.85 (transfer closes most of the shift gap).
- H2: Model C test accuracy >= 0.90 (fine-tuning closes further; Kermany 0.928 remains the bar to BEAT).
- H3: D1 vs D2: if cleaning does NOT improve after transfer, we report that null - the paper's
  claim becomes "the benchmark gap is primarily representation shift, not label noise", which is
  itself the valuable result (judge-agreed framing).
- Beat claim allowed only if: mean test accuracy > 0.928 with the 3-seed t-CI excluding 0.928,
  and at least matching calibration honesty (ECE reported, not cherry-picked).

## Open costs (free-first rule)
TorchXRayVision pip + weights: free. Fine-tune on this 2-CPU box: batch-checkpointed like R-M1.
