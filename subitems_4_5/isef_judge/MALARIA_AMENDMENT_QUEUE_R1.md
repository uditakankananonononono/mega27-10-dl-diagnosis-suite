# Malaria locked amendment queue (PROVIDED ROUND 1, wamid...RkQxOQA=, 11:17:23 IST)

Foldback discipline: critique + NOVELTY CHANGE + commit ref per landed item.
Mapped against science state @283b9cc (post-scrub; science content == 8855788).

## Presentation spine (verdict's headline demands)
A1/A2/W1/W2 LEAD WITH CENSUS, DROP "beats published benchmark" -> replace with
   "at parity under matched protocol (R-M1 0.9477 vs 0.957 bar)". PARTIALLY
   PRIOR-PLANNED: R6 redirection (966d880) already reframed the paper toward
   census-first; this verdict EXTENDS it: the accuracy headline itself moves
   to secondary status. Paper restructure = the R6 novelty-change landing.
A7/A19/W25 12-slide story + one-page summary card -> NEW deliverables
   -> DROPPED per header directive 12:02 IST.
A16 impact section: "what 11.44% noise means for 200+ mid-90s papers" -> NEW.
A14/W15 move 197-record lit audit to appendix (one-paragraph summary in main).
A15/W14 move BBBC 42-accession battery to appendix (portability evidence).
A13 visual summary figure (parity, one-directional noise, flag distribution).
A8 decision tree for dataset users ("what to do about the 249 flags").
W16 reframe dxtool CLI as engineering appendix, not science claim.

## Cheap deliverables first
A20/W4-adjacent flags CSV standalone resource: 249 paths + confidence scores
   -> results/label_noise_flags.csv + README. DO FIRST.
A5/W7/W8 stabilize to 85-image consensus core as primary deliverable;
   architecture-specific flags secondary -> recompute from committed
   cross-check outputs; report Jaccard 0.268 honestly in main text.

## Robustness work (compute)
A4/W9 patient-stratified re-split ("single most important check") ->
   INVESTIGATE FIRST: does cell_images carry patient IDs? (NIH LHBC
   collection metadata; if absent, document honestly and use the strongest
   available proxy; do NOT fabricate stratification.)
A12 patient clustering of flags (same metadata dependency as A4).
A6/W23 R-M1 resize-kernel discrepancy -> PRIOR-PLANNED: PREREG_RESIZE_DIAG
   (acaca5e) running NOW (area fold3 ep4/8; 11 fold-runs left). Decision rule
   locked. This verdict raises it to headline priority.
A17 census on test split too -> NEW run (same protocol as train census).
A18 admissibility-gate sensitivity (thresholds 0.55/0.60/0.65) -> NEW.
A10/W3 other label-noise estimators (CV disagreement, Co-teaching-lite) -> NEW arm.
A9 simulation: expected reported accuracy under 11.44% noise vs mid-90s literature.
A11/W22 flag-type confusion matrix (parasite stage enrichment) + stronger
   forensics (current: only contrast p=2.39e-03, r=0.136).

## Framing/honesty fixes (text)
W3 "11.44% is estimator-derived" -> already stated; STRENGTHEN to abstract.
W5 asymmetry cause not established -> keep both hypotheses, no causal claim.
W6 cleaning-didn't-help -> honest limitation; tie to A17 test-split census.
W10 GCN hybrid adds nothing -> drop "hybrid" framing, report as control arm.
W11 gate thresholds ad hoc -> justify or move to sensitivity (A18).
W12 "first automated" claim -> soften to "no prior found in 197-record screen".
W13 Fuhad reconciliation blocked (dead link) -> keep limitation; try archived
   copy of the audit repo once via web archive.
W17 calibration != census validation -> keep separated.
W18 noise-bound ceiling -> label illustrative.
W19 dose-response subsample n=11023 -> disclose inline.
W20 GCN rate 9.59% below band -> report disagreement openly.
W21 no external malaria cohort -> search for independent dataset (e.g. BBBC041
   malaria?); if none suitable, explicit future-work.
W24 no clinical validation -> keep scope statement prominent.

## Expert adjudication
A3/W4 expert adjudication of 20-30 flags -> PENDING HER RULING (census
   adjudication a/b/c question with parent since 09:01 IST; review card
   results/census_review_card.pdf staged). If no expert available: document
   as future work + strengthen blind-substitute (multi-model consensus).



## A3 EXPERT ADJUDICATION - WAIVED BY OWNER (2026-09-27 11:46 IST)

Owner ruling via WhatsApp 11:46:16 IST (wamid.HBgMOTE4MTM0MDk4NTcxFQIAEhgWM0VCMDczNzU1OEJBNDcxOENGNTUwNQA=,
author=user, verified in channel record): "SKIP CENSUS ..." - the human-expert
adjudication requirement is WAIVED. Proceed without it.
PAPER REQUIREMENT (locked): document in the paper that expert adjudication was
considered and explicitly waived by the owner, and state the residual
label-noise risk honestly (flags remain statistical candidates, not
expert-verified ground truth).



## HEADER DIRECTIVE APPLIED (2026-09-27 12:02 IST)

Fleet-wide, same mega-verdict message (wamid...RURGRAA=, author=user):
"IGNORE ABOUT ISEF DELIVERABLES, IMPROVE PAGE COUNT" - the 12-slide story
and one-page summary card items (A7/A19) are DROPPED. Page count grows with
substance (replication arms, robustness analyses, methods detail) instead.

## Execution order (locked)
1. A20 flags CSV + A5 consensus core recompute (cheap, landed fast).
2. Resize diagnostic completes -> A6/W23 verdict -> R-M2 prereg decision.
3. Paper restructure: A1/A2 spine + A14/A15 appendix moves + W-framing fixes
   (= R6 novelty-change landing, single coherent edit).
4. A4/A12 patient-ID investigation -> stratified re-split if metadata exists.
5. A17/A18/A10/A9/A11 robustness arms.
6. A19/A7 summary card + slide story; A13 figure; A8 decision tree.
