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
Malaria remains one of the highest-burden parasitic diseases worldwide, and
thin-blood-smear microscopy is still the reference diagnostic in most
endemic settings. The NIH Lister Hill National Center \texttt{cell\_images}
collection \cite{rajaraman2018} has become the standard public benchmark for
automated parasite detection in segmented single-cell images. This paper is
the malaria half (item 10.4) of a two-disease deep-learning diagnosis
program; the pneumonia half (item 10.5) appears in a companion paper.
We pursue four goals, in order of priority. (1)~\textbf{Beat the published
benchmark honestly}: train a compact CNN core and a CNN--graph hybrid
(RegionGCN) on the official data with a protocol that prevents any test-set
leakage, and compare against the Rajaraman et al.\ cell-level reference
accuracy whose number we re-derived from the source full text rather than
from secondary citations. (2)~\textbf{Census the label noise}: run a
three-fold out-of-fold confident-learning census over the full training
partition, publish every flagged image identifier, and cross-validate the
noise rate with an independent library. (3)~\textbf{Show the cleaning
effect}: retrain on the census-cleaned data and report the delta, whatever
its sign. (4)~\textbf{Ship a working tool}: the \texttt{dxtool} CLI performs
inference, census, and audit functions on real cells and is evaluated
head-to-head against the strongest external tool we could run on the same
data and split.
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
Pneumonia is the single largest infectious cause of death in children
worldwide, and chest radiography is its workhorse imaging modality. The
Kermany et al.\ \texttt{ChestXRay2017} collection \cite{kermany2018}
(Mendeley Data \texttt{rscbjbr9sj} v2) is the standard public benchmark for
paediatric pneumonia detection in chest X-rays. This paper is the
pneumonia half (item 10.5) of a two-disease deep-learning diagnosis
program; the malaria half (item 10.4) appears in a companion paper.
We pursue four goals, in priority order. (1)~\textbf{Benchmark honesty}:
train a compact CNN core and a CNN--graph hybrid (RegionGCN) under the
official patient-level train/test split, and position the result against
the published Kermany reference and against the strongest external tool we
could run on the \emph{same} test films --- a TorchXRayVision DenseNet-121
evaluated head-to-head on our 624-film official test set.
(2)~\textbf{Census the label noise} with a three-fold out-of-fold
confident-learning census, publish every flagged identifier, and
cross-validate with an independent library. (3)~\textbf{Quantify the
cleaning effect} by retraining on census-cleaned data, reporting the delta
whatever its sign. (4)~\textbf{Ship a working tool}: the \texttt{dxtool}
CLI, benchmarked against the external tool under identical conditions.
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
