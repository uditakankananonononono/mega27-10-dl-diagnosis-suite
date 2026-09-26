# Preregistration: pneumonia improvement ladder (item 10.5)

Locked 2026-09-26, before any new test-set evaluation. The committed best is
test acc 0.8413 (gcn_tuned) vs Kermany et al. 2018 published 0.928 on the same
official 624-image test set. Rule: a rung is evaluated on the test set exactly
once, at its frozen operating point. Failed rungs are reported as negatives.

## Provenance note (verification record)
- Source used in this sandbox: Hugging Face mirror
  `ahulikal/chest-xray-pneumonia-mirror` (public mirror of the Kermany/
  Mendeley chest_xray release), zip sha recorded at download; 5,856 JPEGs,
  official counts (train 5,216 + val 16 + test 624).
- Chain to the original verified release: the previous builder's artifacts
  (census `flagged_ids`, `split.json` indices, label boundary 1349/3883) all
  align on the mirror, and `model_gcn_tuned.pt` evaluated on mirror-rebuilt
  tensors reproduces the committed test probabilities bit-exactly
  (max abs diff 0.0, acc 0.8413). The mirror content is therefore identical
  to the original under the pipeline transform.

## Rung L1: frozen ImageNet features + trained head
- Extractor: torchvision resnet18, ImageNet-1K weights, frozen. Input:
  grayscale 128x128 replicated to 3 channels, ImageNet normalization.
  Features: 512-d penultimate.
- Prior evidence justifying this rung (train-side only): their 2-epoch
  subsampled resnet18 transfer reached val AUC 0.9827 (run_transfer_pneu).
- Head: multinomial logistic regression (L2, C=1) trained on the committed
  train split (4,710 images, seed 42). Val split (522) used only for early
  sanity reporting, not selection.
- Operating point: threshold frozen from 5-fold OOF probabilities of the
  head on the train split (grid 0.30-0.70 step 0.01, maximize accuracy).
  If the OOF-optimal t is within 0.02 of 0.5, t=0.5 (avoid OOF overfit).
- Test evaluation: ONE run on the official untouched 624 test images.
- Success: acc > 0.928. Otherwise: honest negative; escalate to rung L2
  (fine-tune last block + head, preregistered separately before any test eval).
- Discovery arm (unchanged): tiered 91-image consensus core analysis proceeds
  on train-side data only.
