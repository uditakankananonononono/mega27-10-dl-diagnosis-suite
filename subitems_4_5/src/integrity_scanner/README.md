# Benchmark Integrity Scanner

Origin: ChatGPT judge round 3 (isef_judge/pneumonia_round3.md) named this the project's
memorable novelty: not another pneumonia classifier, but a reusable audit that tests whether
a medical-AI benchmark's labels are trustworthy before the benchmark is used to judge models.

- v0.1: report schema + assembly from the two real censuses in this repo
  (pneumonia flag tiers; malaria ours-vs-cleanlab cross-check). Run: python3 scanner.py
  -> results/integrity_report.json
- v1 (planned, prereg-adjacent): end-to-end CLI (images + labels.csv + probe spec ->
  integrity report), then demos on MedMNIST panels. Probe training follows the same
  batch-checkpoint discipline as R-M1.
