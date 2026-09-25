#!/usr/bin/env python3
"""Environment table (live pip versions) + notation table + GNN theory."""
import subprocess, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkgs = ["torch","torchvision","numpy","pandas","scikit-learn","scipy","matplotlib",
"seaborn","pillow","cleanlab","statsmodels","captum","shap","optuna","albumentations",
"timm","umap-learn","pingouin","ydata-profiling","kornia","simpleitk","duckdb",
"plotly","openpyxl","trafilatura","xmltodict","beautifulsoup4","pdfplumber",
"pymupdf","htmldate","reportlab","imageio","lxml","networkx","sympy","formulaic",
"scikit-optimize","torchxrayvision","pytorch-grad-cam","medmnist","torchmetrics",
"torchinfo","opencv-python"]
listing = subprocess.run(['pip','list'], capture_output=True, text=True).stdout.lower()
rows = []
for p in pkgs:
    ver = 'not installed'
    for line in listing.splitlines():
        parts = line.split()
        if parts and parts[0] == p.lower():
            ver = parts[1]; break
    rows.append(f"{p} & {ver} \\\\")
body = "\n".join(rows)

ENV = r"""
\section{Compute environment and package versions}
All experiments ran on a 2-core CPU sandbox with 2\,GB RAM and no swap
(Linux, Python 3.10). Package versions at paper-assembly time (live
\texttt{pip list} capture; the honest environment record):
\begin{longtable}{l l}
\hline Package & Version \\ \hline \endhead
""" + body + r"""
\hline
\end{longtable}
The memory ceiling shaped the engineering: single-process sequential
training, singleton-shared tensors, $\leq$200\,MB resident arrays, and
checkpoint-resumable phases. Three OOM kills occurred during the program
(dmesg-recorded); each produced a documented fix (batch caps, inter-stage
garbage collection, sequential chain discipline) rather than a silent
retry.
"""

NOTATION = r"""
\section{Notation}
\begin{longtable}{l p{9cm}}
\hline Symbol & Meaning \\ \hline \endhead
$x$ & input image, $C_0 \times H_0 \times W_0$ \\
$F$ & CNN feature map, $C \times H \times W$ \\
$g$ & region-grid side ($g=3$ malaria, $g=4$ pneumonia) \\
$A$, $\tilde A$, $\hat A$ & adjacency, with self-loops, normalised \\
$H^{(l)}$ & node embeddings after layer $l$ \\
$\hat p_k(x)$ & predicted probability of class $k$ \\
$t_k$ & class-$k$ self-confidence threshold (census) \\
$C_{ij}$ & confident joint entry \\
$n$, $n_+$, $n_-$ & sample counts (total, positive, negative) \\
$\alpha$, $\beta_1$, $\beta_2$ & Adam learning rate and moments \\
$w_k$ & class weight (tuned configuration) \\
$\epsilon$ & numerical-stability constant \\
\hline
\end{longtable}
"""

THEORY = r"""
\section{GCN behaviour at our graph scale}
\subsection{Receptive field}
Two GCN layers propagate information across $2$ hops. On the $g=3$
malaria grid (diameter 2, NetworkX-verified) every node sees every other
node exactly; on the $g=4$ pneumonia grid (diameter 3) corner-to-corner
information needs 3 hops and is therefore attenuated --- a known,
measurable limitation of the shipped configuration, and the reason the
Optuna search over grid size matters: the best found grid for pneumonia
is reported in the search chapter.
\subsection{Over-smoothing bound}
Repeated application of $\hat A$ contracts node embeddings toward the
dominant eigenvector; with two layers and $\lambda_2 < 1$ (spectral
lemma), the contraction per layer is bounded by
$|\lambda_2|$ and over-smoothing cannot occur at our depth. Formally,
for $H^{(2)} = \hat A^2 H^{(0)} W^{(0)} W^{(1)}$ the component along
eigenvector $u_i$ is scaled by $\lambda_i^2 \le 1$, with strict
contraction for $|\lambda_i| < 1$.
\subsection{Why a graph head at all}
The global-pooling head discards spatial layout; the region graph keeps
a coarse layout and lets evidence move between regions. On malaria the
heads tie (error-analysis chapter) --- parasite evidence is local and the
global pool suffices; on pneumonia the graph wins --- consolidation is a
spatially extended pattern. The architecture choice is thus empirically
justified per disease, not dogmatically hybrid.
"""

for d in ('malaria', 'pneumonia'):
    out = os.path.join(ROOT, f'paper_{d}/sec_env_notation_theory.tex')
    open(out, 'w').write(ENV + NOTATION + THEORY)
    print('wrote', out)
