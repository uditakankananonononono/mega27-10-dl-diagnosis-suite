#!/usr/bin/env python3
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ('malaria', 'pneumonia'):
    extra = (" The TorchXRayVision external tool's curve on the identical films is overlaid (dashed):"
             " it sits far below both shipped models across the whole threshold range, so the"
             " head-to-head win is not an artifact of one operating point."
             if d == 'pneumonia' else
             " The two heads are near-coincident, consistent with the error-analysis chapter:")
    tex = (r"\begin{figure}[h]\centering" + "\n"
           rf"\includegraphics[width=0.62\textwidth]{{../figures/{d}_roc_curves.png}}" + "\n"
           rf"\caption{{ROC curves on the held-out test set, computed from the committed"
           rf" probability dumps (\texttt{{results/{d}/test\_probs\_*.json}}).{extra}}}" + "\n"
           r"\end{figure}" + "\n")
    p = os.path.join(ROOT, f'paper_{d}/sec_rocfig.tex')
    open(p, 'w').write(tex)
    print('wrote', p)
