# ISEF Judge Round 1 - pneumonia project (item 10.5)
Judge: ChatGPT (user's own account, NEW chat 6ab6fc71-b86c-83e8-a414-c0e6604d64ee), sent 04:28 IST 2026-09-26.
Opening block: builder-drafted, agent-verified against committed JSONs before send (155=23+132, est 21.15%, cleanlab 209 superset Jaccard 0.7416, gate 89.81/74.20, tuned 0.9174, exhume19 reconciliation in claim 1).

## Opening (verbatim)

Project: A data-centric audit of a benchmark medical dataset, with small honest models (pediatric pneumonia, Kermany ChestXRay2017)

Summary: Most work on this dataset trains a model and reports test accuracy. We asked a prior question: how noisy are the training labels, can that be shown image by image, and what changes when you clean the data and compare fairly? We ran an automated, gate-enforced label-noise census over the 4,710-image training pool, committed a durable per-image flag list, retrained small from-scratch models on the cleaned data, and benchmarked head-to-head against a strong pretrained off-the-shelf model on the identical official test films.

Claims, each deliberately narrowed:
1. To our knowledge, this is the first AUTOMATED, gate-enforced, per-image-falsifiable label-noise census of ChestXRay2017. Known adjacent work: Kermany's own labels were graded by two expert physicians with a third expert checking only the evaluation set; a 2026 GitHub project (exhume19-coder/radiology-ai-model, April 2026) describes a manual clinical spot-check of 100 images from this dataset asserting about 17% label noise - a manual review of an undocumented 100-image sample with no published per-image flags, no audit code, no gate, and no durable artifact; its figure is asserted, not verifiable; confident-learning tooling (cleanlab) is off the shelf. Our contribution is the gated, durable, per-image artifact plus the downstream evaluation, not the estimator family.
2. The census ESTIMATES 21.15% label noise (996 of 4,710 training images, confident-joint estimate). The durable deliverable is a conservative 155-image flag list (3.29% of 4,710; 23 NORMAL + 132 PNEUMONIA), committed with IDs, and fully contained in cleanlab's own 209-image flag set on the same data (Jaccard 0.742, zero of our flags outside it). We do not present the 155 flags as the noise rate; they are the high-confidence subset.
3. The census had to pass a gate before use: out-of-fold accuracy 89.81% vs a 74.20% majority baseline, admissible under the prespecified rule oof_acc >= max(0.60, majority + 0.05).
4. Cleaning helps small models on the untouched official 624-image test set: baseline CNN 72.12% -> cleaned 79.81% -> tuned 81.73%; baseline GCN 79.01% -> cleaned 84.94% (tuned 84.13%). We do NOT claim to beat Kermany's published 92.8% accuracy on this test set. Our win framing is the audit plus the head-to-head below, not an accuracy race against the dataset authors.
5. Head-to-head on identical test films: torchxrayvision DenseNet-121 (pretrained on large adult chest X-ray corpora, zero-shot) scores 0.799 AUC and 38.46% accuracy at the default 0.5 threshold, which is below the all-pneumonia majority baseline. Our cleaned from-scratch GCN reaches 84.94% accuracy. A strong pretrained adult model fails to operate on this pediatric dataset out of the box.
6. Architecture honesty: the GCN head is a fixed 8-connected grid over the CNN feature map (4g^2-6g+2 undirected edges for a g x g grid). That topology is the standard Moore-neighborhood grid and appears in prior graph-CNN work (SplineCNN 2018, ViG 2022, Patch-GCN 2021, DD-GCN 2020 on chest X-rays, and a January 2026 anatomical-prior chest X-ray graph). We claim the application and the gated evaluation, not a new topology.
7. Weaknesses we state up front: flagged images are sharper and higher-contrast than controls (p = 1.7e-27), so flagging partly tracks image properties; one committed artifact records a failed SimpleITK metric as NaN; the Optuna search was memory-capped (eighth of train, 1 epoch, 4 trials) and served as a relative ranking probe, not a competitor to the full tuned run (committed test AUC 0.9174); ImageNet transfer underperformed our small GCN (69.39% test accuracy) and is committed as a negative result.

Evidence: every number above is in committed JSONs (subitems_4_5/results/pneumonia/*.json on the project repo main branch), including per-image flag IDs, the cleanlab crosscheck, confusion matrices for every model, and the negative results. A one-page fact sheet mapping each number to its file is available.

Ask: Would this win ISEF (or your top regional fair)? Please list weaknesses in priority order, especially anything that would break the narrowed novelty claim, the gate design, or the head-to-head framing.
Show more

## Judge verdict (verbatim)

I will evaluate this as a skeptical ISEF/regional biomedical computing judge, not as a “can this be sold?” pitch. I am not predicting an outcome; I’m assessing what a judge could attack and how much each attack threatens the project.

Short verdict

This is a substantially stronger ISEF-style project than a typical “trained CNN on pneumonia” project. The strongest part is not the model accuracy; it is the data audit + reproducible artifact + downstream consequence analysis. The project has a legitimate scientific question:

Can systematic label-quality auditing of a widely used medical benchmark reveal actionable errors, and does correcting those errors improve small models under controlled evaluation?

That is a fair research question.

However, the highest-risk issues are:

Novelty wording is fragile.

The “label noise” terminology may overclaim what is actually measured.

The head-to-head pretrained model comparison is easy for judges to misunderstand and may distract.

The causal chain “cleaning → improvement” needs stronger ablation evidence.

The dataset itself is heavily benchmarked, so the contribution must be framed as methodology, not discovery.

My estimate: competitive regional → yes; ISEF finalist-level depends heavily on execution of the paper/poster defense. The science is much better than the average ML fair project, but several claims must be tightened.

Priority 1 — The novelty claim is the biggest vulnerability
Current claim:

“first automated, gate-enforced, per-image-falsifiable label-noise census of ChestXRay2017”

This is clever wording, but also dangerous.

A judge may ask:

“How did you establish that nobody else has done this?”

The answer cannot simply be “we searched.”

What could break it:

Someone used cleanlab/confident learning on Kermany before.

Someone performed per-image relabeling internally but did not publish artifacts.

Someone did a similar audit under another name (“dataset quality assessment,” “annotation error analysis,” “data-centric AI”).

The phrase “first” is almost impossible to prove.

Safer framing:

Instead of:

“the first automated label-noise census”

say:

“a reproducible automated audit framework that produces a durable per-image artifact for Kermany ChestXRay2017, with preregistered gating and downstream retraining evaluation.”

That is still novel.

The novelty is not:

cleanlab exists ❌

uncertainty estimation exists ❌

noisy label detection exists ❌

The novelty is:

applying a gated audit pipeline

preserving a falsifiable artifact

measuring downstream model impact

doing fair comparison

That survives.

Risk level: VERY HIGH

Priority 2 — “Label noise” may be scientifically too strong

This is probably the most important methodological attack.

Your 21.15% estimate is not necessarily “wrong,” but judges may challenge:

“How do you know these are mislabeled rather than difficult cases?”

A confident-learning estimate detects likely annotation disagreement, not confirmed clinical error.

Examples:

An image flagged as NORMAL:

could actually contain subtle pneumonia

could be clinically ambiguous

could have borderline findings

could have label based on radiologist interpretation

Therefore:

“label noise” implies ground truth error.

Your evidence proves:

“images inconsistent with learned data distribution and assigned labels”

or:

“candidate annotation-quality issues”

not necessarily:

“incorrect labels.”

Your own wording already helps:

155 flags = high-confidence subset

not presenting flags as true noise rate

Keep that.

I would rename:

“label-noise census” → “label-consistency audit”

or:

“annotation inconsistency audit.”

You can still discuss noise, but define it carefully.

Risk level: VERY HIGH

Priority 3 — The gate design is good, but judges may question circularity

Your gate:

OOF accuracy:
89.81%

versus majority:
74.20%

Requirement:

oof_acc ≥ max(60%, majority + 5%)

This is reasonable.

But a skeptical judge may ask:

“You used a model to decide whether another model can trust the dataset. Isn't this circular?”

The answer:

No, because the gate is not claiming clinical correctness; it only verifies the estimator has predictive signal.

However, you should explicitly say:

“The gate validates that the auditing model is non-random and label-informative; it does not validate medical correctness of individual flags.”

That distinction matters.

Potential improvement:

Add one more gate:

Stability gate

Run:

different random seeds

different model initializations

maybe different architectures

Measure:

How many of the 155 flags persist?

A judge will love:

“The flag set was stable across independent perturbations.”

Currently the gate proves usefulness, not robustness.

Risk level: HIGH

Priority 4 — Cleaning improvement needs stronger causal proof

Your results:

CNN:
72.12 → 79.81 → 81.73

GCN:
79.01 → 84.94

This is promising.

But the attack:

“Maybe you just removed hard images.”

This is the classic cleaning paper problem.

A judge could ask:

“Did cleaning improve because labels became better, or because the dataset became easier?”

You need ablations.

The ideal table:

Training set	Test accuracy
Original	X
Remove random 155 images	X
Remove your 155 images	X
Remove cleanlab 155 images	X
Remove highest-confidence only	X

If your method beats random removal:

that is much stronger evidence.

Currently:

improvement is interesting

causality is plausible

but not fully isolated

Risk level: HIGH

Priority 5 — The DenseNet comparison is interesting but risky

This section has the highest chance of confusing judges.

Your result:

TorchXRayVision DenseNet:

AUC 0.799

accuracy 38.46%

Your GCN:

accuracy 84.94%

A judge might think:

“You compared a pediatric model trained on 4,000 images against a huge adult pretrained model using only threshold accuracy?”

Problems:

Problem 1:

Accuracy at threshold 0.5 is not ideal.

AUC is the better metric.

You report AUC, but then emphasize accuracy.

A reviewer may say:

“Threshold mismatch.”

Problem 2:

Adult vs pediatric domain shift is expected.

The finding is still useful:

"off-the-shelf adult chest X-ray models may not transfer optimally to pediatric benchmarks."

But don't frame as:

"small model beats strong pretrained model."

That sounds like leaderboard gaming.

Better:

"Dataset-specific training outperformed zero-shot adult-domain transfer under this pediatric evaluation."

Much harder to attack.

Risk level: MEDIUM-HIGH

Priority 6 — The 155 image artifact is excellent, but needs clinical validation

This is your biggest missing experiment.

You have:

computational flags ✅

cleanlab agreement ✅

But not:

expert adjudication ❌

A judge may ask:

“How many of your flagged images are actually wrong?”

Without expert review, you cannot quantify precision.

Even a small validation helps.

Example:

Random sample:

50 flagged

50 controls

Have a radiologist/medical student/qualified reviewer classify:

likely correct label

uncertain

likely incorrect

Then report:

“This is not a definitive relabeling, but an audit validation.”

This would massively strengthen the project.

Risk level: HIGH

Priority 7 — Architecture novelty is correctly modest

This part is good.

You explicitly say:

standard Moore-neighborhood grid

Good.

Many student projects fail because they pretend every architecture is novel.

Your framing:

application + evaluation

is defensible.

Do not call it:

novel GCN

new graph topology

new AI architecture

Keep it data-centric.

Risk level: LOW

Priority 8 — The Optuna limitation is handled correctly

Actually, this is a strength.

You explicitly state:

memory cap

small search

not competitor

Judges like negative results.

Do not hide:

ImageNet transfer failed

SimpleITK failed

optimization limited

This makes it look like science rather than a leaderboard chase.

Risk level: LOW

What I would expect judges to ask
Question 1:

“Why should we trust your flags?”

Your answer needs:

“We do not claim they are confirmed errors. They are reproducible candidates for annotation inconsistency. Confirmation requires clinical review.”

Excellent.

Question 2:

“Why does this matter?”

Strong answer:

“Benchmark datasets become infrastructure. If their labels contain systematic inconsistencies, models trained on them inherit those errors. We show that auditing and correction can measurably affect downstream performance.”

Excellent.

Question 3:

“Why not just use a larger model?”

Answer:

“The goal was not maximizing accuracy. It was isolating the effect of data quality under controlled small-model conditions.”

Excellent.

Changes I would make before ISEF submission
Must do
1. Remove “first”

Replace with:

“a reproducible automated audit framework”

2. Replace “label noise census”

Use:

“annotation consistency audit”

3. Add one ablation:

Random removal vs flagged removal.

This is probably the single highest-value experiment.

4. Add a stability experiment:

Different seeds → overlap of flags.

5. Tone down DenseNet language.
Overall judge scorecard (not a prediction)
Category	Assessment
Scientific question	Strong
Dataset importance	Strong
Reproducibility	Very strong
Novelty	Moderate-strong if reframed
ML sophistication	Moderate
Experimental rigor	Strong but needs ablation
Medical validity	Moderate
Risk of overclaiming	Currently high
Presentation potential	Very high

The project’s ceiling is not determined by adding a bigger neural network. The ceiling is determined by whether you can convince judges that the artifact is trustworthy and the improvement is caused by better data rather than easier data.

If you fix the novelty language and add the removal/stability ablations, this becomes a much more defensible ISEF-level data-centric AI project.
