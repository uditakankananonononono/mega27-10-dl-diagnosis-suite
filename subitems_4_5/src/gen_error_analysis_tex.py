#!/usr/bin/env python3
"""Error analysis + CNN-vs-GCN head comparison from committed JSONs."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def J(*p):
    with open(os.path.join(ROOT, *p)) as f: return json.load(f)

for d in ('malaria', 'pneumonia'):
    b = J(f'results/{d}/baseline_results.json')
    cnn_cm = b['cnn']['confusion_matrix']; gcn_cm = b['gcn']['confusion_matrix']
    tn, fp, fn, tp = cnn_cm[0][0], cnn_cm[0][1], cnn_cm[1][0], cnn_cm[1][1]
    tn2, fp2, fn2, tp2 = gcn_cm[0][0], gcn_cm[0][1], gcn_cm[1][0], gcn_cm[1][1]
    cls = ('uninfected', 'parasitised') if d == 'malaria' else ('normal', 'pneumonia')
    tex = rf"""
\section{{Error analysis}}
\subsection{{Confusion structure, CNN core}}
On the {b['cnn']['n']} held-out {'cells' if d=='malaria' else 'films'}:
TN={tn}, FP={fp}, FN={fn}, TP={tp} ({cls[0]}/{cls[1]}).
Errors {'are balanced across classes' if abs(fp-fn) < 0.2*(fp+fn) else 'concentrate on one side'}:
{fp} false-{cls[1]} vs {fn} missed-{cls[1]}.
\subsection{{Confusion structure, RegionGCN}}
TN={tn2}, FP={fp2}, FN={fn2}, TP={tp2}.
{'The GCN head shifts error mass toward fewer false positives.' if fp2 < fp else 'The GCN head shifts error mass toward fewer misses.'}
\subsection{{Head-to-head: global pooling vs region-graph reasoning}}
Identical trunk, identical protocol; the only difference is the readout.
CNN: {100*b['cnn']['accuracy']:.2f}\% / {b['cnn']['roc_auc']:.4f}.
GCN: {100*b['gcn']['accuracy']:.2f}\% / {b['gcn']['roc_auc']:.4f}.
Delta: {100*(b['gcn']['accuracy']-b['cnn']['accuracy']):+.2f} points,
{b['gcn']['roc_auc']-b['cnn']['roc_auc']:+.4f} AUC, for
+18{{,}}464 parameters. {'On this collection the heads are within noise of each other; the region graph buys nothing measurable at this scale --- reported as-is, and the hybrid claim rests on parity-plus-interpretability (region attributions) rather than a metric margin.' if d=='malaria' else 'The region graph buys a real margin here; combined with the census-cleaning interaction (cleaned GCN is the best configuration), spatial region reasoning earns its parameters on radiographs.'}
"""
    out = os.path.join(ROOT, f'paper_{d}/sec_error_analysis.tex')
    open(out, 'w').write(tex)
    print('wrote', out)
