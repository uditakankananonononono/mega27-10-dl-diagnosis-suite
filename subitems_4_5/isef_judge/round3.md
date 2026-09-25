# ISEF Judge Round 3 - malaria project (item 10.4)
Judge: ChatGPT (user's own account, chat 6ab6a0d2-d9c4-83e9-b475-f82b35bbeb81), 2026-09-25 ~22:17-22:23 IST.
Topic: Fuhad per-image flag list unrecoverable (dead repo, archive holds no flags); asked what substitutes for image-level validation, whether the durability finding strengthens the contribution, and for the updated top-3 weaknesses.

## Question (verbatim)

Round 3 update on your #1 weakness (external validation of the 11.44% estimate via the Fuhad overlap). We attempted the flag-by-flag overlap experiment and hit a documented dead end that we now report as a finding in itself: Fuhad's correction list lives at their reference [41], github.com/gracelynxs/malaria-detection-model - that URL is now 404, and the Internet Archive's last capture (2022-01-31) shows the repo contained only README.md, build_dataset.py and train_model.py - no correction list, no per-image flags. The prior audit's flags are unrecoverable, so the overlap is impossible as published. Our paper now documents the dead pointer and the archive inspection instead of implying the comparison was run, and makes the methodological point explicit: an audit whose flags are not durably committed cannot be checked, reused, or reconciled - our 249 flags are committed, hashed, and regenerable. Given that the overlap is physically impossible (their flags no longer exist anywhere), what should we do instead to satisfy the validation requirement? And does the durability finding itself strengthen the project's contribution in your judgment? Please also give your updated top-3 remaining weaknesses.
Show more

## Judge answer (verbatim)

Yes—but I would make one important distinction.

The dead-link finding strengthens the reproducibility/methodological contribution, but it does not itself validate the 11.44% estimate. You should replace the missing Fuhad-overlap validation with independent validation experiments that test different aspects of your estimator.

I also checked the literature again. Fuhad et al. explicitly report the 647/750 expert-removal procedure and cite the gracelynxs/malaria-detection-model repository as reference 70. 
PubMed Central (PMC)
+1
 Later papers continue to describe the 26,161-image corrected dataset, so the fact that the audit occurred is independently documented even though the per-image correction artifact may no longer be recoverable. 
PubMed Central (PMC)
+1

What I would do instead
1. Build a synthetic-label-noise recovery experiment

This is probably your best replacement.

Start with a subset whose labels you treat as the reference labels, then inject known amounts of controlled label noise:

1%

2.5%

5%

10%

15%

20%

Do both:

symmetric noise

deliberately directional noise

Then run your exact census pipeline without telling it the injected noise level.

The crucial question becomes:

Does the estimator recover the known noise rate and direction?

For example:

True injected noise	Estimated noise
2%	...
5%	...
10%	...
15%	...
20%	...

You can calculate:

bias

RMSE

calibration

coverage of your confidence intervals

sensitivity to directionality

This validates the estimator itself, rather than depending on an unrecoverable historical flag list.

It is not a substitute for expert biological validation—but it answers an important question that Fuhad cannot answer anymore:

Does the method work when we know the answer?

2. Use Fuhad's published aggregate numbers as a consistency check, not ground truth

You still have a useful piece of information:

1,397 images were removed: 647 + 750.

That is a published aggregate result, even though the individual identities are unavailable. 
PubMed Central (PMC)

You can therefore explicitly say:

"An image-level reconciliation was attempted but could not be performed because the cited repository no longer contains the correction artifact. We therefore do not claim agreement with Fuhad at the image level."

Then compare only what is legitimately comparable:

Fuhad: 1,397 manually identified/suspicious images

Your estimator: 3,153 estimated flags, if 11.44% × 27,558 is what your estimate represents

Fuhad direction: 647 vs 750

Yours: 2,358 vs 164

But do not calculate an apparent disagreement as though the sets are directly comparable.

And be especially careful about interpreting 647:750 versus 2,358:164. Those numbers may represent fundamentally different constructs.

That discrepancy is actually a research question.

3. Your strongest validation experiment may be cross-method agreement

You already have an independent-architecture Jaccard of 0.841.

Push this much harder.

Don't merely say:

"Jaccard = 0.841."

Construct a stability analysis:

Architecture A

→ estimated flags

Architecture B

→ estimated flags

Architecture C

→ estimated flags

Different seeds

→ estimated flags

Then report:

pairwise Jaccard

flag-count variation

directional ratio

rank correlation of suspicion scores

If completely different models converge on approximately the same suspicious images, that is powerful evidence that your finding isn't merely one network's idiosyncrasy.

Still not proof of biological mislabeling—but strong evidence of methodological robustness.

4. Do a "negative control" experiment

This could be excellent for your admissibility gate.

Create datasets where you know the estimator should not be trusted:

shuffled labels

random/uninformative features

collapsed classifier

near-random classifier

deliberately inadequate OOF model

Your desired result:

The estimator refuses to produce a census.

Then demonstrate that if you disable the gate, the same conditions can produce apparently plausible but meaningless noise estimates.

That gives you a very clean scientific result:

The gate is not merely a software convenience; it controls a measurable failure mode of the estimator.

This may ultimately be more defensible than trying to prove every individual real-world flag is genuinely mislabeled.

5. Do an independent human validation sample if feasible

You don't need all 3,153 flags.

A pre-specified random sample of flagged and unflagged images reviewed by a qualified expert would be tremendously valuable.

For example, conceptually:

random flagged sample

random unflagged sample

blinded review

same criteria

report agreement

The important thing is that the reviewer shouldn't know whether the image was flagged.

That gives you an actual estimate of:

P(true annotation problem | algorithm flag)

and

P(algorithm flag | expert assessment).

That is much stronger than another ML model agreeing with yours.

If expert review isn't available, don't manufacture a substitute—just make the limitation explicit.

Does the durability finding strengthen the contribution?

Yes, but as a secondary methodological contribution.

I would not claim:

"Fuhad's study is not reproducible."

That's too broad.

The published scientific result remains accessible, and other papers still describe the 647/750 correction. 
PubMed Central (PMC)
+1

Your narrower and much stronger observation is:

"The published audit's aggregate findings remain citable, but the per-image correction artifact required for automated flag-level reconciliation is no longer available at the cited location."

That's a reproducibility observation.

And your contrast is legitimate:

Their audit:

publication → aggregate result → external artifact no longer sufficient for image-level reconciliation

Your audit:

algorithm → exact filenames → hashes → committed evidence → regeneration procedure

That is worth highlighting.

But don't let it become the main scientific discovery. A judge could easily respond:

"That's a software/reproducibility lesson, not evidence that your 11.44% estimate is correct."

They're right.

So I'd frame it as:

Primary contribution

Automated estimation of directional label noise with an admissibility criterion.

Secondary contribution

Durable, image-level audit artifacts enabling future reconciliation.

Reproducibility finding

Historical aggregate conclusions may remain available even when their underlying image-level audit artifact is no longer recoverable.

That's a very defensible hierarchy.

Updated TOP 3 weaknesses
#1 — The 11.44% / 14:1 estimate still lacks a real-world ground-truth validation

This remains the biggest weakness.

The Fuhad comparison is now impossible at image level, and you've correctly documented that.

So you need to replace it with:

known-noise synthetic recovery + cross-model stability + ideally blinded expert review.

If you do those three, this weakness drops dramatically.

#2 — Your estimator's assumptions could still generate systematic bias

This is now more important than the missing Fuhad list.

A judge should be able to ask:

"What assumptions does confident learning make here, and how do you know they hold for malaria-cell morphology?"

You need an explicit assumption/sensitivity analysis.

Change:

architecture

calibration

folds

thresholds

seed

preprocessing

class balance

and demonstrate what happens to the estimated 11.44%.

If it ranges from, say, 7% to 16%, then 11.44% shouldn't be presented as a precise number.

If it remains around 11–12%, that's compelling.

#3 — The project still needs stronger evidence about the consequence of the noise

You've shown that:

noisy benchmark → 96.08%

and

your cleaned version → 95.50%.

But the causal interpretation is still incomplete.

You want repeated experiments demonstrating:

Does benchmark label noise materially change model conclusions?

That's more interesting than whether your CNN gets 96%.

Ideally demonstrate:

performance across noise-removal levels

multiple seeds

patient-level confidence intervals

possibly calibration

external dataset performance

Then you can answer:

"Does the benchmark's label corruption merely exist, or can it actually alter scientific conclusions drawn from the benchmark?"

That is the question I would want your poster to end with.

My revised ISEF judgment

The project has changed substantially since Round 1.

Round 1

Novelty vulnerability: 🔴 major

Round 2

Novelty vulnerability: 🟢 substantially resolved

Round 3

Validation vulnerability: 🟡/🟠 still the central issue

But now you have a very clean path forward:

Known-noise recovery → cross-model stability → blinded expert sample → downstream consequence analysis.

If you execute those properly, I would stop worrying about whether this is "novel enough."

I'd start worrying about something more favorable:

whether the data support the strength of the conclusions you're making.

And that's exactly the kind of problem you want to have at this stage.

One final wording change I'd strongly recommend: don't call the Fuhad artifact "physically impossible" in the paper. Say "not reproducibly recoverable from the cited public source as of [date], despite archival inspection." That's more precise and much harder for a reviewer to challenge.
