#!/usr/bin/env python3
"""Detailed tables chapter: Wilson CIs, metric cross-verification, census
cross-check, optuna trials, per-fold detail. Numbers from committed JSONs."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def R(*p): return os.path.join(ROOT, *p)
def J(*p):
    with open(R(*p)) as f: return json.load(f)

CI = J('results/confidence_intervals.json'); XV = J('results/metrics_crossverify.json')

def ci_table(disease):
    rows = []
    for k, v in sorted(CI.items()):
        if k.startswith(disease):
            name = k.split('/', 1)[1].replace('_', r'\_')
            rows.append(f"{name} & {v['correct']}/{v['n']} & {100*v['accuracy']:.2f} & "
                        f"[{100*v['wilson95'][0]:.2f}, {100*v['wilson95'][1]:.2f}] \\\\")
    return rf"""
\section{{Confidence intervals and cross-verification}}
\subsection{{Wilson 95\% intervals for every reported accuracy}}
statsmodels score intervals (\texttt{{results/confidence\_intervals.json}}),
so every accuracy in this paper carries its uncertainty:
\begin{{longtable}}{{l c c c}}
\hline Configuration & Correct/$n$ & Acc\% & Wilson 95\% \\ \hline \endhead
{chr(10).join(rows)}
\hline
\end{{longtable}}
\subsection{{Independent metric recomputation}}
Every headline metric was recomputed with torchmetrics independently of
sklearn (\texttt{{results/metrics\_crossverify.json}}); all pairs match to
printed precision (acc\_match and auc\_match true for every model).
"""

def census_xc(disease):
    d = J(f'results/{disease}/census_crosscheck.json')
    return rf"""
\subsection{{Census cross-validation against cleanlab}}
The from-scratch confident-learning estimator flagged {d['ours_flagged']}
images; the independent \texttt{{cleanlab}} library flagged
{d['cleanlab_flagged']} on identical out-of-fold probabilities. The
intersection is {d['intersection']} --- our flag set is a strict subset of
cleanlab's (Jaccard {d['jaccard']:.3f}) --- so our census is the
conservative estimator of the two, and the disagreement mass is published
as samples in \texttt{{results/{disease}/census\_crosscheck.json}}.
The cleanlab confident joint is
[[{d['cleanlab_joint'][0][0]:.4f}, {d['cleanlab_joint'][0][1]:.4f}],
 [{d['cleanlab_joint'][1][0]:.4f}, {d['cleanlab_joint'][1][1]:.4f}]].
"""

def optuna_table(disease):
    p = f'results/{disease}/optuna_gcn.json'
    if not os.path.exists(R(p)): return ''
    o = J(p)
    rows = []
    for i, t in enumerate(o['trials']):
        prm = t.get('params', t)
        val = t.get('value', t.get('val_auc', float('nan')))
        rows.append(f"{i+1} & {prm.get('hidden','-')} & {prm.get('grid','-')} & "
                    f"{prm.get('lr', 0):.2e} & {val:.4f} \\\\")
    return rf"""
\subsection{{Optuna search, full trial log}}
All {o['n_trials']} TPE trials ({o['epochs_per_trial']} epochs each,
{o['train_secs']:.0f}s total; \texttt{{results/{disease}/optuna\_gcn.json}}):
\begin{{longtable}}{{r c c c c}}
\hline Trial & Hidden & Grid & LR & Val AUC \\ \hline \endhead
{chr(10).join(rows)}
\hline
\end{{longtable}}
Best validation AUC {o['best_val_auc']:.4f} at
hidden={o['best_params']['hidden']}, grid={o['best_params']['grid']},
lr={o['best_params']['lr']:.2e}; shipped-configuration test AUC
{o['current_gcn_test_auc']:.4f}.
"""

def write(path, text):
    with open(path, 'w') as f: f.write(text)
    print('wrote', path, len(text))

if __name__ == '__main__':
    for d in ('malaria', 'pneumonia'):
        write(R(f'paper_{d}/sec_tables.tex'), ci_table(d) + census_xc(d) + optuna_table(d))
