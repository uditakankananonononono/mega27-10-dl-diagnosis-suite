# ISEF Judge Loop - Malaria (item 10.4) - Round 2
Chat: https://chatgpt.com/c/6ab6a0d2-d9c4-83e9-b475-f82b35bbeb81
After: Fuhad 2020 verified at PMC7277980; claims narrowed to first AUTOMATED census; reconciliation section added (commit 0a48f33).

## Question (verbatim)
Follow-up on your Round 1 critique. We verified your novelty challenge and you were right: Fuhad et al. 2020 (Diagnostics 10(5):329, PMID 32443868) did a manual expert audit of the same 27,558-image dataset, setting aside 647 parasitized + 750 uninfected images (reducing it to 26,161). It was even inside our own 197-record literature screen - invisible because its title says nothing about labels. We have now: (1) rewritten the claim everywhere as "the first AUTOMATED, estimator-based, per-image-falsifiable census" - explicitly standing on Fuhad, not against it; (2) added a reconciliation section: their removal was balanced 647:750 at 5.07%, while our statistical estimate is 11.44% and one-directional 2358:164; both readings stated (asymmetric visual confusability vs model bias; an independent-architecture crosscheck at Jaccard 0.841 supports the flagged subset); (3) committed in writing to a flag-by-flag overlap experiment against their published correction list, marked honestly as in-progress rather than implied done; (4) corrected the literature-screen claim to title/metadata precision, noting the miss. Re-judge honestly: does this fix the novelty problem? What are now the TOP 5 remaining weaknesses, and would you now predict a win at ISEF?
Show more

## Answer (verbatim)
Yes. I would now consider the novelty problem substantially fixed. More importantly, the correction actually makes the project scientifically stronger because you now have a prior expert audit to use as an independent benchmark rather than something you have to pretend doesn't exist.

I would not call the novelty issue completely closed until the overlap experiment is actually completed, but the conceptual problem is gone.

Fuhad et al. really did manually identify/remove 647 parasitized and 750 uninfected images from this dataset. 
PubMed Central (PMC)
 Your claim is now appropriately narrower: you're proposing an automated estimator-based census, rather than claiming discovery of the underlying labeling problem. That's a legitimate distinction.

And CBIO is a very appropriate ISEF category for this kind of work: Society for Science defines it specifically around computational/mathematical techniques applied to biological systems. 
Society for Science

My revised verdict

Would I predict a win? No.

But my answer has changed from:

"There is a potentially fatal novelty problem."

to:

"This is now a legitimately high-end ISEF project whose remaining weaknesses are primarily validation and inference, not novelty."

That's a substantial upgrade.

I would now put the project in the serious finalist / potentially Grand-Award-competitive tier if the remaining experiments come out well. I still would not predict a Grand Award because ISEF judging is comparative: your project is judged against the other finalists in the category, and presentation/interview are 35 of the 100 points. 
Society for Science

The TOP 5 remaining weaknesses
1. The 11.44% estimate still needs external validation

This is now #1 by far.

You have a statistical estimator saying:

11.44%

and Fuhad's expert/manual procedure effectively gives:

5.07% removed.

Those aren't directly contradictory because they're measuring different things. But a judge is going to ask:

"Why should I trust 11.44%?"

Your Jaccard = 0.841 cross-architecture result is encouraging, but it establishes stability, not truth.

That's the distinction I'd hammer.

Stability ≠ validity.

Two models can agree because they share the same bias.

Your planned flag-by-flag comparison against Fuhad is therefore extremely important.

You want to know:

Of Fuhad's 1,397 images, how many does your method flag?

What is precision?

What is recall?

How does overlap differ by direction?

What does your method identify beyond Fuhad?

Are those additional candidates qualitatively/statistically different?

If your method recovers most of the expert list and produces additional coherent candidates, your 11.44% estimate becomes substantially more credible.

If it only recovers a small fraction, that's a serious problem.

Do not cherry-pick this experiment.

Whatever happens, show it.

2. The 14:1 directional result could still be model-induced

This is now your most exciting result and your most vulnerable result simultaneously.

You have:

2,358 vs 164

That's enormous.

But the causal interpretation needs restraint.

You currently have several possible explanations:

A

There really is strongly asymmetric annotation error.

B

The visual distinction between the categories produces asymmetric model uncertainty.

C

The training labels induce asymmetric classifier behavior.

D

The confident-learning estimator has directional bias under these class/morphology distributions.

E

Some combination of A–D.

Your sentence:

"The contamination is one-directional"

is therefore too strong.

I'd use:

"The estimated label noise is strongly directional, with parasitized-labeled images accounting for approximately 14× as many estimated cross-label errors as uninfected-labeled images."

That's extremely strong while remaining scientifically defensible.

Then attack it.

The killer experiment:

Run the directional estimate across:

CNN architecture 1

CNN architecture 2

RegionGCN

independent classifier

multiple random seeds

perhaps a non-neural statistical classifier

Then report the ratio distribution.

If you get something like:

12.7×, 14.4×, 13.6×, 15.1×...

that's much harder to dismiss as a model artifact.

3. Your 96.08% result is still secondary—and potentially vulnerable

This is actually less important than you may think.

You don't want the judge leaving your booth thinking:

"Another malaria CNN."

There are already many deep-learning malaria-classification papers.

Your scientifically interesting result is the benchmark audit.

The CNN should function as an experimental instrument supporting the investigation.

Also, your 96.08% versus published 94.0% comparison must be handled carefully.

Unless you reproduce the competing method under the same split, preprocessing, patient partition, and evaluation protocol, don't call it simply:

"We beat the published benchmark."

Say:

"Our model achieved 96.08% under our patient-level evaluation protocol; published results on the benchmark report approximately 94%, although protocols are not necessarily identical."

Even better:

Reproduce the baseline yourself.

Then:

Reproduced baseline: X
Ours: 96.08%

Same data. Same patient split. Same test set.

That is much more convincing.

4. The "noise as regularization" finding is not yet a finding

This remains one of the places where I would be most aggressive during the interview.

You currently have:

Original = 96.08%
Cleaned = 95.50%

You can conclude:

Removing the estimated noise did not improve performance in this experiment.

You cannot yet conclude:

Noise acts as a regularizer.

That's a mechanistic explanation.

And there are many possible alternatives.

The beautiful way to resolve this is to make it a real experiment.

Train across multiple independently generated seeds:

Training labels	Seed 1	Seed 2	...	Mean
Original				
Fuhad-cleaned				
Your-cleaned				

Then progressively remove labels:

0%, 2%, 5%, 8%, 11.44%, 15%

If you see a reproducible curve, now you have something interesting.

You might discover:

benchmark noise does not monotonically damage predictive performance.

That is a much more interesting result than merely saying "my model got 96%."

5. Your statistical unit needs to be patients, not images

This is the technical weakness I'd expect a serious computational-biology judge to notice.

You have thousands of images, but they're nested within patients.

Therefore, saying:

"n = 2,758"

can be misleading if you're treating all 2,758 cells as independent biological observations.

Your effective biological sampling structure is:

patients → cells/images

For confidence intervals and hypothesis testing, I would want to see patient-level resampling.

For example:

sample patients with replacement → include all their cells → recompute metric → repeat.

Then give patient-clustered bootstrap CIs.

This matters especially for:

accuracy

AUC

sensitivity/specificity

comparisons between models

Mann–Whitney analyses

If your current CIs bootstrap individual images, I would redo them.

One thing I would now change in your framing

Your project has accidentally become more interesting than the original version.

Originally it sounded like:

"I discovered mislabeled malaria images and built a small CNN."

Now the actual scientific question is closer to:

"Can an automated, admissibility-controlled statistical framework audit the reliability of a widely used biomedical benchmark, quantify directional label noise, and determine whether benchmark corruption actually affects downstream model performance?"

That is a much better ISEF question.

And it naturally produces three independent scientific results:

Result 1 — Measurement

How much estimated label noise exists?

Result 2 — Structure

Is that noise directional?

Result 3 — Consequence

Does correcting it actually change predictive performance?

The CNN is then the experimental instrument for Result 3.

That's coherent.

What would make me substantially more confident about a Grand Award?

There are five concrete boxes I'd want checked:

□ 1. Fuhad overlap completed

You need the actual confusion matrix:

Fuhad positive / yours positive
Fuhad positive / yours negative
Fuhad negative / yours positive
Fuhad negative / yours negative

Plus precision/recall/F1.

□ 2. Directionality survives independent architectures

Not just:

Jaccard = 0.841

but:

directional ratio remains large across independent models/seeds.

□ 3. Patient-level CIs

Especially for the 96.08% / 0.9929 results.

□ 4. Multi-seed noise-removal experiment

Turn the 95.50 vs 96.08 observation into a statistically interpretable experiment.

□ 5. External validation

A completely independent malaria-image dataset would be extremely valuable.

And there's one more thing I'd worry about at ISEF: the interview

Your project is now sophisticated enough that your ability to explain it may become a bigger determinant than adding another 0.3% accuracy.

ISEF explicitly evaluates whether the finalist understands the science, interpretation, limitations, independence, and future research—not merely whether the project produced impressive numbers. 
Society for Science

A top judge might ask you, rapidly:

"Why isn't Fuhad's 5.07% contradictory to your 11.44%?"

You should immediately answer:

"They're estimating different quantities. Fuhad manually identified images they considered sufficiently suspicious to remove; my estimator produces a model-dependent estimate of latent label noise. I therefore don't treat 11.44% as ground truth. The purpose of our overlap experiment is to quantify how the automated estimate relates to the expert-audited reference."

That's an excellent answer.

Then:

"Why does 14:1 prove contamination?"

You:

"It doesn't prove causation. It demonstrates strongly asymmetric estimated label disagreement. We are testing whether that asymmetry persists across independent architectures and seeds before making a stronger claim."

Excellent.

Then:

"Why did removing noise hurt?"

You:

"It didn't establish that noise is beneficial. It established that removing our estimated noisy examples did not improve this model under this protocol. We're testing whether that observation survives repeated seeds and graded removal."

Excellent.

That style of answer is exactly what you want.

My revised scorecard

Not an official ISEF score—just my assessment of the scientific position:

Dimension	Current assessment
Research question	Very strong
Originality	Strong after reframing
Technical sophistication	Very strong
Statistical sophistication	Strong, with patient-level issue to fix
Reproducibility	Exceptional potential
Biological significance	Moderate–strong
Model contribution	Moderate
Audit contribution	Very strong
External validity	Currently weak
Main scientific vulnerability	Validity of 11.44% / 14:1 estimates
Interview potential	Extremely high
ISEF competitiveness	Potentially very high

ISEF's actual science rubric gives 10 points to the research question, 15 to design/methodology, 20 to execution/statistics, 20 to creativity/impact, and 35 to presentation/interview. 
Society for Science
 Your revised project has the ingredients to score well across essentially all five—but the unresolved validation questions sit directly in the 20-point execution/statistics category.

So, would I predict a win?

No—not honestly.

But I'd now say:

This is no longer a project I'd expect to lose because its central novelty claim collapses under literature review. It is a project that could plausibly contend for a top CBIO result if the remaining validation experiments confirm the estimator and directional finding.

And there is a very specific threshold I'd use:

If Fuhad-overlap is strong + 14:1 survives independent architectures + patient-level statistics are clean + repeated noise-removal experiments are convincing, I would regard this as genuinely Grand-Award-caliber on scientific substance.

If those experiments fail, the project can still be good—but the central claim becomes much less compelling.

One final strategic point: don't hide the Fuhad correction. Put it in the project. A sophisticated judge may actually respect the fact that you discovered your original novelty claim was wrong, corrected it, and turned the prior work into a validation benchmark. That is much closer to how real scientific research works than pretending the literature search was flawless.
