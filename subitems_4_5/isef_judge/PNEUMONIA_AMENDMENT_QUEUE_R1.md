# Pneumonia locked amendment queue (PROVIDED ROUND 1, wamid...Q0Y3RgA=, 11:19:16 IST)

Foldback discipline: critique + NOVELTY CHANGE + commit ref per landed item.
Mapped against pneumonia arm of build 283b9cc (science content == 8855788).

## Presentation spine
A1/W1 LEAD WITH 42:1 DIRECTIONALITY, not the cleaning gain -> abstract/intro
   reorder: "one-directional contamination (973:23 = 42:1) co-varying with
   acquisition quality"; cleaning gain demoted (random-removal control matches).
A2/W2/W15 DROP "beats external tool" framing -> replace with domain-shift
   finding: "adult-pretrained CXR models fail zero-shot on paediatric films
   (38.5% vs 84.9%)"; add threshold-recalibration caveat inline.
A7/A19/W14 12-slide story + one-page summary card -> NEW deliverables.
A16 impact section ("what 21.15% noise means for 150+ mid-90s papers").
A14/W14-adjacent move 150-record lit audit to appendix (one-paragraph summary).
A15/W10/W19 move MedMNIST atlas to appendix (portability evidence; report the
   2.4x PneumoniaMNIST-vs-ChestXRay2017 discrepancy as an open question).
A13 visual summary figure (class distribution, 42:1 noise, flag distribution,
   sharpness stratification).
A8 decision tree for ChestXRay2017 users ("what to do about the 155 flags").

## Cheap deliverables first
A20 flags CSV standalone: 155 paths + confidence scores -> DO FIRST (paired
   with malaria A20 in one commit).
A6 stabilize to 91-image consensus core as primary deliverable; lower-
   confidence flags secondary.

## Mechanism / robustness work
A4/W20 report-parsing hypothesis DIRECT TEST (highest-value cheap win):
   compare flagged films vs their parsed reports; show flagged PNEUMONIA
   films have equivocal/absent report text. CHECK FIRST: does the
   ChestXRay2017-derived collection we hold include report text? (kermany
   dataset is images-only; original Guangzhou dataset had labels only -
   verify; if no reports, document honestly, use filename/metadata proxies.)
A5/W7 blurred-NORMAL control: are blurred NORMAL films also flagged?
   Distinguishes mislabeled vs hard-to-label readings.
A9 simulation: expected reported accuracy under 21.15% noise vs literature.
A10 other label-noise estimators (CV-disagreement arm).
A11 flag-type confusion matrix (bacterial vs viral enrichment).
A12/W11 patient clustering of flags + census fold stratification check
   (patient IDs availability same investigation as malaria A4).
A17 census on test split.
A18 admissibility-gate sensitivity (0.55/0.60/0.65).
W17 recalibration arm for deployed models (ECE 0.125/0.141 -> isotonic).
W16 no independent pneumonia cohort -> search (e.g. RSNA pneumonia); else
   explicit future work.
W3 92.8% reference cross-regime/paywalled -> keep as secondary-source
   caveat; do not headline.
W9 GCN advantage vs threshold imbalance -> isolate via threshold-matched
   comparison before claiming spatial benefit.
W18 noise-bound ceiling -> label illustrative.

## Expert adjudication
A3/W5 expert adjudication of 20-30 flags -> PENDING HER RULING (pneumonia
   adjudication question with parent since 09:02 IST; connects to malaria A3;
   no radiologist available to us - if declined: future work + multi-model
   consensus substitute).

## Execution order (locked)
1. A20 flags CSV + A6 consensus core (with malaria equivalents, one commit).
2. A5 blurred-NORMAL control + A4 report-text availability check.
3. Paper restructure: A1/A2 spine + appendix moves + W-framing fixes.
4. A12/A17/A18/A10/A9/A11 robustness arms.
5. A19/A7 summary card + slides; A13 figure; A8 decision tree.
