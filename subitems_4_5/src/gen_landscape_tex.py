#!/usr/bin/env python3
"""Methods-landscape comparison + expanded formulas per disease."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAL = r"""
\section{Methods landscape and positioning}
\begin{table}[h]\centering\small
\begin{tabular}{p{4.2cm} p{2.6cm} r r r l}
\hline
Method & Regime & Params & Input & Acc.\ \% & Verified? \\
\hline
Rajaraman custom CNN (feature-extractor regime) & cell-level & --- & variable & 94.0 & source, double-extracted \\
Rajaraman VGG-16 extractor & cell-level & 138M & 224px & 94.5 & source, double-extracted \\
Rajaraman patient-level & patient-level & --- & --- & 95.9 & source, double-extracted \\
timm ResNet-18 linear probe (ours, this paper) & cell-level & 11M (frozen) + probe & 224px & 91.84 & measured, committed JSON \\
\textbf{CNN core (this paper)} & cell-level & \textbf{140k} & 48px & \textbf{96.08} & measured, committed JSON \\
\textbf{RegionGCN (this paper)} & cell-level & \textbf{159k} & 48px & \textbf{95.98} & measured, committed JSON \\
\hline
\end{tabular}
\caption{Verified-numbers-only landscape. Secondary-citation numbers that
could not be source-verified are excluded by rule; ``---'' means the
source does not state the value.}
\label{tab:landscape}
\end{table}
Three positioning facts emerge. (i) The compact from-scratch models beat
both the published extractor regime and a modern pretrained probe at two
to three orders of magnitude fewer parameters. (ii) The pretrained probe
\emph{underperforms} the published extractor numbers despite a stronger
backbone --- feature extraction on this collection saturates, and
task-trained small models win. (iii) The residual accuracy ceiling is
plausibly label-noise-bound: with 11.44\% contaminated training labels,
a mid-90s ceiling is consistent with every verified number in the table,
including the reference's own.
"""

PNE = r"""
\section{Methods landscape and positioning}
\begin{table}[h]\centering\small
\begin{tabular}{p{4.6cm} p{2.4cm} r r r l}
\hline
Method & Regime & Params & Input & Acc.\ \% & Verified? \\
\hline
Kermany Inception-v3 transfer (published) & official split & 24M & 299px & 92.8 & paywalled; 2 corroborating sources \\
TorchXRayVision DenseNet-121 (same films) & official split & 8M & 224px & 38.5 & measured, committed JSON \\
\textbf{CNN core, cleaned (this paper)} & official split & \textbf{140k} & 128px & \textbf{79.81} & measured, committed JSON \\
\textbf{RegionGCN, cleaned (this paper)} & official split & \textbf{159k} & 128px & \textbf{84.94} & measured, committed JSON \\
\hline
\end{tabular}
\caption{Verified-numbers-only landscape. The Kermany number is
cross-regime (see the verification chapter); the TorchXRayVision row is
the same-split like-for-like comparison.}
\label{tab:landscape}
\end{table}
The honest reading: (i) on identical films and split, our compact models
beat the strongest runnable external tool by +46.5 points --- the
like-for-like tool-quality claim. (ii) The published 92.8\% stands
cross-regime; the sandbox-budget transfer shot (69.39\%) shows the gap
does not close cheaply, and we do not claim it does. (iii) The
cleaning-driven +5.9-point gain and the 21.15\% census estimate suggest a
large share of the remaining gap is label-side, not model-side: decontaminated
retraining at scale is the highest-value follow-up the data supports.
"""

FORM2 = r"""
\section{Further formulas}
\subsection{Matthews correlation coefficient}
\begin{equation}
\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\end{equation}
reported as the single-number balanced summary where class imbalance
makes accuracy misleading.
\subsection{Precision--recall AUC}
\begin{equation}
\mathrm{AP} = \sum_k (R_k - R_{k-1})\, P_k ,
\end{equation}
the average-precision step approximation to the precision--recall curve.
\subsection{Calibration (expected calibration error)}
With $B$ bins over confidence,
\begin{equation}
\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}
\left| \mathrm{acc}(b) - \mathrm{conf}(b) \right| .
\end{equation}
\subsection{Bayes error under label noise}
If labels flip with class-conditional rates $\epsilon_{ij}$, the observed
class conditionals mix:
\begin{equation}
\tilde p(y = j \mid x) = \sum_i \epsilon_{ij}\, p(y = i \mid x),
\end{equation}
so any achievable accuracy is bounded by the mixture's Bayes rate ---
the formal statement behind the `label-noise-bound ceiling' claim.
\subsection{Bootstrap confidence interval}
For statistic $T$ and resamples $b = 1 \dots B$,
\begin{equation}
\mathrm{CI}_{1-\alpha} = \big[ T^{*}_{(\alpha/2)},\; T^{*}_{(1-\alpha/2)} \big],
\end{equation}
the percentile interval used alongside the Wilson intervals.
"""

for d, land in (('malaria', MAL), ('pneumonia', PNE)):
    out = os.path.join(ROOT, f'paper_{d}/sec_landscape.tex')
    open(out, 'w').write(land + FORM2)
    print('wrote', out)
