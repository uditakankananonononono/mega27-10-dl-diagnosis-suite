# Label-noise flag resources (provided-verdict items malaria A20, pneumonia A20)

Produced 2026-09-27 from the committed census outputs. Expert adjudication:
considered and explicitly WAIVED by the owner (WhatsApp 11:46:16 IST,
wamid.HBgMOTE4MTM0MDk4NTcxFQIAEhgWM0VCMDczNzU1OEJBNDcxOENGNTUwNQA=). All
flags remain STATISTICAL CANDIDATES, not expert-verified ground truth.

## label_noise_flags.csv (A20 standalone resource)
One row per flagged image: malaria 249, pneumonia 155 (exactly the committed
census flag sets in results/<disease>/label_noise_census.json).
Columns:
- rank_within_disease: rank by noise_score, descending.
- disease, sample_id, true_label.
- p_labeled_oof: out-of-fold probability assigned to the image's given label
  (from results/<disease>/oof_probs_train.json, aligned via
  results/<disease>/split.json train indices).
- noise_score = 1 - p_labeled_oof (higher = more suspect).

## label_noise_cleanlab_agreement.csv (robustness cross-check)
IDs on which our census and an independent cleanlab run AGREE
(results/<disease>/census_crosscheck.json agreed_ids): malaria 249/249 flags
covered at Jaccard 0.841, pneumonia 155/155 at Jaccard 0.742. This is the
estimator-agreement tier, NOT the cross-architecture consensus core.

## Not yet in this release
- Cross-architecture (CNN x GCN) consensus core (verdict malaria A5: 85-image
  core; pneumonia A6: 91-image core): requires a GCN OOF-probability run
  (only GCN test probs are committed). Queued as its own compute item; the
  id list will land here as label_noise_consensus_core.csv when computed.
