# Interview prep - judge-scripted Q&A (items 10.4 malaria + 10.5 pneumonia)
Compiled verbatim from the ChatGPT judge rounds (isef_judge/round4.md,
isef_judge/pneumonia_round1.md). These are the judge's suggested answers in
its own words - learn the substance, then say it your way.

## The one question that now matters most (malaria judge, R4)
"Why should I believe your flags represent annotation errors rather than
difficult examples?"
Answer (judge-scripted): "They are not equivalent. We report them as
estimated label-noise candidates, not confirmed errors. We validated
estimator behavior under known corruption, and we show architecture-dependent
uncertainty at the image level. The strongest reproducible finding is the
population-level noise signal and directional asymmetry; individual flags
require external adjudication."

## Pneumonia judge (R1) expected questions
1. "Why should we trust your flags?"
   "We do not claim they are confirmed errors. They are reproducible
   candidates for annotation inconsistency. Confirmation requires clinical
   review."
2. "Why does this matter?"
   "Benchmark datasets become infrastructure. If their labels contain
   systematic inconsistencies, models trained on them inherit those errors.
   We show that auditing and correction can measurably affect downstream
   performance."
3. "Why not just use a larger model?"
   "The goal was not maximizing accuracy. It was isolating the effect of
   data quality under controlled small-model conditions."

## The closing answer both judges converged on
"Why should the biomedical ML community care that this benchmark is noisy?"
"Because benchmark accuracy can hide annotation uncertainty, and we provide
a reproducible framework to audit and quantify that uncertainty."

## How to tell the findings (malaria judge R4 restructure - now reflected in the paper)
1. The benchmark has ~10-11% estimated label noise (malaria 11.44%, band 9.90-11.48%; pneumonia 21.15%).
2. The noise is strongly asymmetric (malaria 14:1; pneumonia 42:1).
3. A cross-model consensus core (malaria: 85 images flagged by BOTH CNN and GCN auditors) is the highest-confidence audit output.
4. Individual flags outside the core are representation-dependent candidates, tiered honestly in the artifact.

## If asked "is it a first?"
Never claim a first. Say: "We screened 150+ records individually and found
no prior automated per-image audit of these collections; the closest prior
work is manual (Fuhad et al. 2020 for malaria; a 100-image MD spot-check on
GitHub for pneumonia). We reconcile against both in the paper."

## If asked about real-world use
Malaria models are well calibrated on the test set (ECE 0.02/0.009) - a
predicted 0.9 is right about nine times in ten on average. Say "the
probabilities are meaningful in aggregate"; never promise per-patient
certainty. Pneumonia compact
models are only moderately calibrated (ECE ~0.13) - say so; the honest
limitation lands better than a dodge.

## Known soft spots and the honest line
- No expert adjudication yet: "The flags are candidates; expert blind
  review of a flag/control sample is the scoped next step."
- Cleaning gain causality (pneumonia): random-removal ablation shows
  flagged removal beats random removal under identical protocol - labels,
  not ease.
- Gate circularity: "The gate certifies the auditor is non-random and
  label-informative; it does not certify the medical truth of any flag."
