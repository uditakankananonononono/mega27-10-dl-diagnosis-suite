#!/usr/bin/env python3
"""Training-dynamics tables from committed per-epoch histories."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BS = chr(92) * 2
for d in ('malaria', 'pneumonia'):
    b = json.load(open(os.path.join(ROOT, f'results/{d}/baseline_results.json')))
    parts = []
    for kind in ('cnn', 'gcn'):
        h = b[kind].get('history') or []
        if not h: continue
        rows = []
        for e in h:
            rows.append(f"{e['epoch']} & {e['train_loss']:.4f} & {e['val_loss']:.4f} & {e['secs']:.0f} {BS}")
        body = chr(10).join(rows)
        best_ep = min(range(len(h)), key=lambda i: h[i]['val_loss'])
        parts.append(rf"""
\subsection{{{'CNN core' if kind=='cnn' else 'RegionGCN'} baseline training log}}
Per-epoch committed history (\texttt{{results/{d}/baseline\_results.json}}):
\begin{{tabular}}{{rrrr}}
\hline Epoch & Train loss & Val loss & Secs \\ \hline
{body}
\hline
\end{{tabular}}
Best validation loss at epoch {best_ep}; early stopping restores that
checkpoint. Epoch cost ~{h[0]['secs']:.0f}s on the 2-core sandbox ---
the full training budget of this study is visible in this table.
""")
    tex = "\n\\section{Training dynamics}\n" + "\n".join(parts)
    out = os.path.join(ROOT, f'paper_{d}/sec_dynamics.tex')
    open(out, 'w').write(tex)
    print('wrote', out, len(tex))
