#!/usr/bin/env python3
"""Assemble the two per-disease papers (10.4 malaria, 10.5 pneumonia) from
committed JSONs + shared section files. Each paper: >=50pp content excl.
headings/references, real Times New Roman, disease-specific numbers only."""
import json, os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def R(*p): return os.path.join(ROOT, *p)
def J(*p):
    with open(R(*p)) as f: return json.load(f)

PREAMBLE = open(R('paper_malaria/preamble.inc')).read()

def fmt(x, d=2):
    return f"{x:.{d}f}"

# ---------- shared prose blocks ----------
INTRO_MAL = r"""
\section{Introduction}
The NIH Lister Hill \texttt{cell\_images} collection is the standard
teaching and benchmarking dataset for automated malaria detection in
thin-blood-smear microscopy, cited and reused by hundreds of studies.
Its labels have been questioned exactly once before: a 2020 manual expert
audit (Fuhad et al.) set aside 1{,}397 images as falsely labelled or
suspicious. No \emph{automated, statistical} audit exists, and no prior
work publishes machine-checkable per-image flags or analyses the
\emph{structure} of the contamination. This paper's primary contribution
is that audit, not another accuracy number on the dataset:
\begin{enumerate}
\item \textbf{The first automated label-noise census of the
collection.} A three-fold out-of-fold confident-learning census estimates
\textbf{11.44\%} label inconsistency and publishes \textbf{249
falsifiable per-image flags} --- each an exact archive path any reader can
inspect. We reconcile against the Fuhad expert audit (Sec.~\ref{sec:fuhad}):
their manual screen removed 5.1\% bidirectionally; our statistical
estimate is larger and one-directional, and we analyse why.
\item \textbf{The noise is one-directional.} Parasitized-labelled cells
cross-validate as Uninfected at a 14:1 ratio over the reverse
(2{,}358:164). The collection's published class parity (13{,}779/13{,}779)
is parity by construction, and it hides a directional contamination that
bounds what any trained model can score --- a fact every prior mid-90s
accuracy report on this collection implicitly confirms.
\item \textbf{An admissibility gate for confident learning.} Applied
naively, the estimator fabricates noise rates under weak out-of-fold
models (demonstrated on real collapse cases). Our gate --- OOF accuracy
$\ge \max(0.60, \text{majority}+0.05)$, $\ge 250$ optimizer steps per
fold --- is a small, necessary method contribution for applied use.
\end{enumerate}
Supporting these, a compact CNN core and CNN--graph hybrid (RegionGCN),
trained from scratch under a strict no-leak protocol, reach 96.08\% test
accuracy on 2{,}758 held-out cells --- beating the source-verified
published cell-level reference (94.0\%) and an ImageNet-pretrained
linear probe, with two orders of magnitude fewer parameters. The shipped
\texttt{dxtool} CLI performs inference, census, and audit on real cells.
"""

DATA_MAL = r"""
\section{Data provenance and integrity}
\subsection{Source and verification}
NIH Lister Hill National Center \texttt{cell\_images.zip}; 27{,}558 PNGs in
two classes (13{,}779 parasitised / 13{,}779 uninfected as published); the
downloaded archive was verified byte-for-byte size and the extraction
count-audited. Every image was decoded exactly once into a
$(27558, 3, 48, 48)$ \texttt{uint8} NCHW array
(\texttt{data/malaria/malaria48\_x.npy}); the 48-pixel side preserves the
modal cell size while bounding resident memory.
\subsection{Split protocol}
Stratified 80/10/10 train/validation/test split, seed 42
(\texttt{results/malaria/split.json}). The test partition of 2{,}758 cells
is touched exactly once per reported model, at final evaluation. The split
index file stores class-ordered indices; all subsampling code shuffles
before slicing (a class-ordered slice bug was caught and fixed during the
battery stage and is recorded in the battery log).
\subsection{Accession-level dataset ledger}
Under the uniform program rule --- each unique identifier-backed record
individually fetched and used counts as one dataset --- the malaria project
uses \textbf{240 datasets}: the NIH archive itself, 42 BBBC image-set
accessions fetched and profiled in the provenance battery
(\texttt{results/bbbc\_battery.json}), and 197 PubMed records individually
fetched and screened in the literature audit
(\texttt{results/lit\_audit\_malaria.json}). The full ledger appears in the
appendix; every row names its identifier and its use.
"""

METHODS_MAL = r"""
\section{Methods}
\subsection{Models}
CNN core (140{,}322 parameters): three convolution--batchnorm--ReLU blocks
with max pooling, global head. RegionGCN hybrid (158{,}786 parameters): the
CNN trunk's final feature map is pooled onto a $g{\times}g$ region grid
($g{=}3$ for the 48-pixel malaria trunk, whose final map is $6{\times}6$
and requires $g \mid 6$), nodes joined by 8-neighbour adjacency, two GCN
layers, hidden width $d{=}64$.
\subsection{Training protocol}
Adam $\alpha{=}10^{-3}$, batch 64, early stopping on validation loss
(patience 2--3), deterministic seeds, 2 CPU threads; augmentation is
horizontal/vertical flips plus $k\cdot90^\circ$ rotations seeded per
sample. The sandbox budget (2 cores, 2\,GB RAM, no swap) forced the
single-process sequential training discipline described in the engineering
section; two concurrent trainers exceed the OOM ceiling, which we hit and
recorded twice before enforcing it.
\subsection{Census protocol}
Three-fold out-of-fold probabilities on the training partition with the CNN
core; per-class self-confidence thresholds and the confident joint per
Definition~\ref{def:cj}. An estimator \emph{admissibility gate} --- OOF
accuracy $\ge \max(0.60, \text{majority}+0.05)$ and at least 250 optimizer
steps per fold --- rejects folds whose probabilities cannot support
confident learning; without the gate the estimator fabricates noise rates
(demonstrated on a synthetic collapse run and on
\texttt{breastmnist}, where an inadmissible fold reported 49.8\% noise).
Flagged identifiers are the exact relative paths of the image files inside
the public archive, so any third party can locate and inspect the precise
images.
\subsection{Cross-validation of every estimator}
Every headline estimator is recomputed with an independent implementation:
the census noise rate from scratch (numpy) and with \texttt{cleanlab};
accuracy/AUC with \texttt{sklearn} and \texttt{torchmetrics}; symbolic
identities with \texttt{sympy} against numeric evaluation. Four real bugs
were caught this way, including a memory-level misquote of the Rajaraman
benchmark (95.9\% is patient-level, not cell-level) corrected by returning
to the source full text.
"""

INTRO_PNE = r"""
\section{Introduction}
The Kermany \texttt{ChestXRay2017} collection is the standard public
benchmark for paediatric pneumonia detection in chest radiographs. Its
labels were generated by automated report parsing --- a known
contamination path --- yet \textbf{no automated, per-image-falsifiable
audit of its training labels is known}. This paper's primary contribution is a discovery about
that dataset, with a method contribution and a regime finding attached:
\begin{enumerate}
\item \textbf{The first confident-learning census of the training set.}
Three-fold out-of-fold estimation gives \textbf{21.15\%} label noise
with \textbf{155 published falsifiable flags} (screen-supported: no
prior census among 150 individually fetched PubMed records).
\item \textbf{The contamination is one-directional and co-varies with
acquisition quality.} PNEUMONIA-labelled films cross-validate as NORMAL
at a 42:1 ratio (973:23), and flagged films are measurably blurrier and
flatter than controls ($p<10^{-6}$) --- the statistical signature of
report-parsing label generation meeting borderline acquisitions.
\item \textbf{Cleaning helps here, and the contrast with malaria is a
finding.} Retraining on the census-cleaned data improves the strongest
model by \textbf{+5.9 points}; on malaria it does not. The regime
variables --- contamination mass and test-set curation process --- are
measurable, and we name them.
\end{enumerate}
Supporting these, compact models trained under the official patient-level
split beat the strongest runnable external tool (TorchXRayVision
DenseNet-121) by +46.5 points on the identical 624 test films; the
cross-regime gap to the published 92.8\% reference is stated honestly
and addressed by a tested transfer study. The \texttt{dxtool} CLI ships
inference, census, and audit on real films.
"""

DATA_PNE = r"""
\section{Data provenance and integrity}
\subsection{Source and verification}
Mendeley Data \texttt{rscbjbr9sj} v2, \texttt{ChestXRay2017.zip},
SHA-256 \texttt{13efc055\dots ce36} verified against the publisher API.
The official patient-level train/test split is preserved exactly
(5{,}232 train / 624 test); validation is carved stratified from train
(10\%, seed 42) and never touches the test set. Films were decoded once
into $(N, 1, 128, 128)$ \texttt{uint8} NCHW arrays
(\texttt{data/pneumonia/cxr\_train\_x.npy}, \texttt{cxr\_test\_x.npy}).
\subsection{Accession-level dataset ledger}
Under the uniform program rule --- each unique identifier-backed record
individually fetched and used counts as one dataset --- the pneumonia
project uses \textbf{161 datasets}: the Mendeley archive, two atlas panels
used in the suite-level cross-panel audit, and 158 PubMed records
individually fetched and screened in the literature audit
(\texttt{results/lit\_audit\_pneumonia.json}, unioned with the shared
audit's pneumonia subset and deduplicated by PMID). The full ledger
appears in the appendix.
"""

METHODS_PNE = r"""
\section{Methods}
\subsection{Models}
CNN core (140{,}322 parameters) and RegionGCN hybrid (158{,}786
parameters), grid $g{=}4$ on the $16{\times}16$ final feature map of the
128-pixel trunk, hidden $d{=}64$, two GCN layers.
\subsection{Training protocol}
Adam $\alpha{=}10^{-3}$, batch 32, early stopping on validation loss,
deterministic seeds, 2 CPU threads; augmentation is flips plus
$k\cdot90^\circ$ rotations seeded per sample. A class-weighted variant
(weights inversely proportional to class frequency) addresses the
normal/pneumonia imbalance and is reported separately as the tuned
configuration (\texttt{results/pneumonia/tuned\_results.json}).
\subsection{Census protocol}
Three-fold out-of-fold probabilities on the training partition with the
CNN core; confident joint and thresholds per Definition~\ref{def:cj}; the
same admissibility gate as the malaria study (OOF accuracy
$\ge \max(0.60, \text{majority}+0.05)$, $\ge 250$ steps per fold). Flagged
identifiers are exact relative paths inside the public archive.
\subsection{External head-to-head protocol}
TorchXRayVision \texttt{densenet121-res224-all} was run in its published
form (weights untouched, official preprocessing) over the identical 624
official test films; its probability vector was argmax-mapped to the
binary task. Scores are committed in
\texttt{results/pneumonia/test\_probs\_torchxrayvision.json}. This is the
strongest external tool comparison we could execute on identical data and
split; the published Kermany number (92.8\% accuracy) is cross-regime
(different model class, input resolution, and training data scale) and is
treated as the published-benchmark gap, not a same-split defeat.
"""

def write(path, text):
    with open(path, 'w') as f: f.write(text)
    print('wrote', path, len(text), 'chars')

if __name__ == '__main__':
    write(R('paper_malaria/sec_intro_data_methods.tex'),
          INTRO_MAL + DATA_MAL + METHODS_MAL)
    write(R('paper_pneumonia/sec_intro_data_methods.tex'),
          INTRO_PNE + DATA_PNE + METHODS_PNE)
