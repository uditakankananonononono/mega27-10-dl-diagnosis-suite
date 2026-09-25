#!/usr/bin/env python3
"""Dataset card sections (collection anatomy from committed manifests)."""
import json, os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAL = r"""
\section{Dataset card: NIH cell\_images}
\subsection{Anatomy}
27{,}558 segmented single-cell PNGs, class parity by construction
(13{,}779 parasitised / 13{,}779 uninfected as published; extraction
count-audited byte-for-byte). Cells are centred crops from thin-smear
slides (193 patients per the source record), Giemsa stain, variable crop
size around a modal ~48px.
\subsection{Tensorisation}
Decoded once to $(27558, 3, 48, 48)$ uint8 NCHW
(\texttt{data/malaria/malaria48\_x.npy}); BILINEAR resize; independent
imageio re-decode of 60 sampled cells matches the stored array at 0 max
absolute pixel difference (\texttt{results/tool\_battery\_3.json}).
Identifiers preserved in \texttt{malaria48\_ids.npy} so every array row
maps to its archive path.
\subsection{Known contamination}
The census estimates 11.44\% label inconsistency with a strongly
one-directional pattern (Parasitized$\to$Uninfected cross-validation
dominates 2{,}358:164). Downstream users should treat the parity of the
published class counts with suspicion: parity by construction does not
imply purity.
"""

PNE = r"""
\section{Dataset card: ChestXRay2017}
\subsection{Anatomy}
5{,}856 paediatric chest radiographs (5{,}232 train / 624 test,
patient-level official split preserved exactly); labels from automated
report parsing per the source publication; frontal (AP/PA) views.
\subsection{Tensorisation}
Decoded once to $(N, 1, 128, 128)$ uint8 NCHW arrays
(\texttt{data/pneumonia/cxr\_train\_x.npy}, \texttt{cxr\_test\_x.npy});
independent imageio re-decode of 40 sampled films matches at 0 max pixel
difference. Identifiers preserved for every row.
\subsection{Known contamination}
The census estimates 21.15\% label inconsistency, one-directional
(PNEUMONIA$\to$NORMAL cross-validation dominates 973:23), consistent with
report-parsing label generation (a film whose report emphasises follow-up
or support devices can carry a pneumonia label with clear lungs).
Flagged films are measurably blurrier and flatter than controls
($p < 10^{-6}$ sharpness), so acquisition quality and label noise
partially co-vary in this collection.
"""

for d, t in (('malaria', MAL), ('pneumonia', PNE)):
    out = os.path.join(ROOT, f'paper_{d}/sec_datacard.tex')
    open(out, 'w').write(t)
    print('wrote', out)
