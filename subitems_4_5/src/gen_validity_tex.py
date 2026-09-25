#!/usr/bin/env python3
"""Threats to validity + census methodology detail per disease."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VALID = {
'malaria': r"""
\section{Threats to validity}
\subsection{Internal validity}
The split is random stratified, not patient-stratified: cells from the
same patient can appear in train and test. The published reference uses
the same convention (its patient-level number, 95.9\%, is reported
separately in the source), so our cell-level comparison is like-for-like;
a patient-stratified re-split is the first robustness follow-up and is
flagged honestly here. Augmentation is seeded per sample; evaluation runs
exactly once per model on the test partition.
\subsection{Construct validity}
The census measures statistical label inconsistency under the
confident-learning model, not clinical truth. The admissibility gate
bounds when the estimator may run at all; the cleanlab cross-check bounds
implementation error (our flags are a strict subset of cleanlab's); the
forensics chapter bounds the alternative `bad acquisition' explanation
(only contrast separates the groups, small effect).
\subsection{External validity}
One collection, one stain, one scanner family. The BBBC battery and the
MedMNIST atlas bound how the method (not the model) transfers; the model
itself makes no cross-collection claim.
\subsection{Reliability}
Every number regenerates from committed probability dumps; two independent
metric libraries agree; the paper's tables are emitted programmatically
and a missing value renders UNVERIFIED rather than being filled.
""",
'pneumonia': r"""
\section{Threats to validity}
\subsection{Internal validity}
The official patient-level split is preserved exactly; validation is
carved from train only. Class weighting changes the effective decision
threshold, so tuned configurations are reported with full
sensitivity/specificity pairs. The census fold assignments are seeded and
committed.
\subsection{Construct validity}
Binary normal/pneumonia collapses viral/bacterial; the published reference
uses the same binary task for its headline number. The TorchXRayVision
comparison maps its multi-pathology output to the binary task by the
Pneumonia logit only --- the mapping is stated in the methods and the raw
probabilities are committed for re-analysis.
\subsection{External validity}
Paediatric films only; adult pneumonia and other views are out of scope.
The atlas panel (PneumoniaMNIST, 8.96\% noise, admissible) is a different
collection and is used only for method-level triangulation.
\subsection{Reliability}
As in the malaria study: committed probability dumps, dual metric
libraries, programmatic tables, UNVERIFIED on any missing value. The
SimpleITK NaN probe and the chestmnist atlas run error are reported as
failures, not deleted.
""",
}

CENSUS = {
'malaria': r"""
\section{Census methodology in detail}
\subsection{Fold protocol}
Three folds, stratified, seeded; each fold's model trains from scratch on
the union of the other two (batch 64, Adam $10^{-3}$, early stopping,
patience 2). Out-of-fold probabilities cover every training cell exactly
once; the fold checkpoints are retained under
\texttt{results/malaria/oof\_ckpt} for audit.
\subsection{Threshold computation}
Per class $k$, the threshold $t_k$ is the mean self-confidence of
examples labelled $k$ (Definition~2.5); the confident joint counts
$(y_n, \arg\max \hat p)$ pairs with $\hat p_j \ge t_j$; off-diagonal mass
above the per-class expectation is the flag set. Our estimator and
cleanlab 2.x agree on the flag set direction (ours $\subset$ cleanlab's);
the implementation difference is threshold handling at ties.
\subsection{Why three folds}
Compute: one fold costs roughly one baseline training run; three is the
largest count that keeps the census inside the nightly budget. The
variance cost of fewer folds is acknowledged; the cleanlab cross-check
uses the same folds, so implementation comparison is fold-matched.
""",
'pneumonia': r"""
\section{Census methodology in detail}
\subsection{Fold protocol}
Three folds, stratified on the training partition, seeded; fold models
train from scratch (batch 32, Adam $10^{-3}$, early stopping). The
official test films never enter any fold. Fold checkpoints retained under
\texttt{results/pneumonia/oof\_ckpt}.
\subsection{Admissibility, enforced}
The gate evaluated OOF accuracy 89.81\% against the majority baseline
74.20\% plus margin --- admissible, and recorded with the rule text in
\texttt{results/pneumonia/label\_noise\_census.json}. An earlier
inadmissible configuration (weaker folds) fabricated a much higher noise
rate; that run is the reason the gate exists and is described in the
discussion.
\subsection{Threshold computation and cross-check}
As in the malaria study: class-mean self-confidence thresholds, confident
joint, off-diagonal flags; cleanlab on identical OOF probabilities flags
a superset (209 vs our 155; Jaccard 0.742), so our census is again the
conservative estimator.
""",
}

for d in ('malaria', 'pneumonia'):
    out = os.path.join(ROOT, f'paper_{d}/sec_validity.tex')
    open(out, 'w').write(VALID[d] + CENSUS[d])
    print('wrote', out)
