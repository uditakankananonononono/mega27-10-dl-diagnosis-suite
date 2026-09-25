# Pneumonia judge Round 2 question (DRAFT TEMPLATE - fill [..] from committed JSONs after ablations land; same chat as R1)
Your Round 1 must-dos, now executed:
1. "first" removed everywhere - the paper now reports absence of prior art from the 150-record screen and names closest prior art itself (Fuhad manual audit for malaria; exhume19-coder/radiology-ai-model manual 100-image MD spot-check for this collection, verified at source: ~15% FN on NORMAL, ~20% FP on PNEUMONIA).
2. Terminology: "label-noise census" -> "annotation-consistency audit"; flags -> "candidate annotation-quality issues" throughout.
3. Random-removal ablation RUN: removing 155 RANDOM training films (3 seeds, identical split/config/test) gives CNN [mean+-sd]% / GCN [mean+-sd]% vs flagged-removal CNN 79.81% / GCN 84.94% (baseline 72.12/79.01) - [verdict sentence].
4. Seed-stability of the flags RUN: census re-run at seeds 7/123 gives [n] and [n] flags, overlap with canonical 155 = [x]/[y] (Jaccard [j1]/[j2]); [k]-image all-seed consensus core published as highest-confidence tier.
5. DenseNet reframed as zero-shot adult-domain transfer: AUC-led (+0.139/+0.148), plus NEW calibration evidence - the transfer model's probabilities are worse than the base-rate prior on these paediatric films (ECE 0.6126, Brier skill -1.58) vs our tuned models (ECE 0.125/0.141, skill 0.43/0.48).
Also added: gate-circularity defense (gate certifies the auditor is non-random/label-informative, not medical truth of flags).
Question: with these executed, what weaknesses remain before this is a defensible ISEF-level data-centric AI project? Bind each to a concrete fix; say explicitly when a Round 1 weakness is closed.
