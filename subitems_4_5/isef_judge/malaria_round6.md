# Malaria judge round 6 - RULE 6 redirection after R-M1 preregistered FAIL

- Model/surface: ChatGPT free web (config-c), fresh conversation 2026-09-27 00:37-00:54 IST
- Conversation: https://chatgpt.com/c/6ab81860-03c4-83e8-83d0-3185a93b5fa0
- Paste: full updated 58pp malaria paper (32 parts incl. R-M1 verdict section) + redirection question
  (verbatim question in this file's QUESTION section; full response: malaria_round6_capture.txt)
- Trigger: R-M1 matched-protocol rung FAILED (mean 0.94771 < 0.957 bar, 95% t-CI [0.9226,0.9728];
  VERDICT.json @ 39b5331). Negatives are never terminal; stuck negative goes to the judge for redirection.

## QUESTION (verbatim)
That was the complete malaria paper. CONTEXT: the preregistered matched-protocol rung R-M1 vs
Rajaraman 2018 just FAILED (5-fold mean 0.94771 < 0.957 bar, 95% t-CI [0.9226,0.9728]; folds
0.9361/0.9641/0.9544/0.9357/0.9482, AUCs 0.977-0.992; full protocol in the R-M1 section you just
read). Per the project rules a stuck negative goes to you for redirection. QUESTIONS: (1) Judge the
current paper as an ISEF-style judge: 3 biggest weaknesses now. (2) Name the single strongest
concrete pivot for rung R-M2 that could plausibly clear the 0.957 matched-protocol bar - among or
beyond: eliminating the preprocessing resize-kernel difference, a stronger backbone under identical
protocol, per-fold seed ensembling for variance reduction, or reframing the comparison on AUC. Say
what evidence would prove it works. (3) The single most novel addition that would make this project
memorable beyond the benchmark chase. Be concrete and critical, not encouraging.

## RESPONSE: malaria_round6_capture.txt (verbatim, 6608 chars)

## DECISION (agent, verifying judge claims independently where load-bearing)
1. FIRST run the cheap resize-kernel diagnostic the judge ranked highest information-per-hour:
   same CNN + folds, preprocessing = {PIL resize, OpenCV INTER_AREA, INTER_LINEAR, Rajaraman-matched
   pipeline}. If accuracy moves >1pp, preprocessing is a confounder for the R-M1 comparison. Judge's
   "0.253 confidence shift" claim traces to our own paper's robustness battery (internally grounded).
2. THEN lock PREREG_R-M2.md BEFORE any R-M2 eval: stronger backbone (ResNet-18 class, from scratch,
   no pretrained weights, no external data) under the exact R-M1 matched protocol; primary endpoint
   5-fold mean acc > 0.957 with all folds reported, paired fold comparison vs R-M1, bootstrap CI.
3. NOVELTY CHANGE (this round's landed change): paper reframing per judge weakness 1 - the project
   stops leading with the benchmark-beat claim; the label-noise census/audit tooling becomes the
   centerpiece contribution, plus the human-in-the-loop label-uncertainty atlas as the named
   discovery direction. Landed as: R-M1 section already published (9de2edc) + paper abstract/intro
   reframing edit (this commit or next) + PREREG_R-M2.md.
4. AUC reframing explicitly REJECTED (goalpost move; preregistered metric stands).
