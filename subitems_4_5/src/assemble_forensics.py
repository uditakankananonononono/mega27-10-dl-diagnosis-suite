#!/usr/bin/env python3
"""Flagged-image forensics chapter, honest per-disease statistics."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def R(*p): return os.path.join(ROOT, *p)
def J(*p):
    with open(R(*p)) as f: return json.load(f)

def sec(disease):
    d = J(f'results/{disease}/flagged_image_properties.json')
    f, c, mw = d['flagged'], d['control'], d['mann_whitney']
    rows = []
    for prop in ('sharpness', 'contrast', 'entropy'):
        rows.append(f"{prop} & {f[prop]['mean']:.4f} & {f[prop]['median']:.4f} & "
                    f"{c[prop]['mean']:.4f} & {c[prop]['median']:.4f} & "
                    f"{mw[prop]['U']:.0f} & {mw[prop]['p']:.2e} \\\\")
    table = (r"\begin{table}[h]\centering\small" + "\n"
        r"\begin{tabular}{lcccccc}" + "\n\hline" + "\n"
        r"Property & Flag mean & Flag med & Ctrl mean & Ctrl med & $U$ & $p$ \\"
        + "\n\hline\n" + "\n".join(rows) + "\n\hline\n"
        r"\end{tabular}" + "\n"
        rf"\caption{{Flagged vs control image properties ({disease}; Mann--Whitney, pingouin). "
        rf"Data: \texttt{{results/{disease}/flagged\_image\_properties.json}}.}}"
        r"\label{tab:forensics}" + "\n" + r"\end{table}" + "\n")
    if disease == 'malaria':
        verdict = rf"""
\subsection{{Verdict, stated honestly}}
Only contrast separates the malaria flagged set from controls
($p={mw['contrast']['p']:.2e}$); sharpness ($p={mw['sharpness']['p']:.2f}$)
and entropy ($p={mw['entropy']['p']:.2f}$) do not. The malaria flags are
therefore \emph{{not}} a low-quality-image artifact: they are
label-consistent-statistics outliers in otherwise normal-looking cells,
which strengthens the interpretation that the census found genuine
label-side inconsistency rather than a acquisition-quality stratum. An
earlier draft claim that flagged images are generically atypical was
corrected against this table and survives only for pneumonia.
"""
    else:
        verdict = rf"""
\subsection{{Verdict}}
All three properties separate the pneumonia flagged set from controls ---
sharpness decisively ($p={mw['sharpness']['p']:.2e}$), contrast
($p={mw['contrast']['p']:.2e}$), entropy ($p={mw['entropy']['p']:.2e}$).
Flagged films are measurably atypical: they are blurrier and flatter than
unflagged films. Two readings are compatible with the data: (i) genuinely
mislabeled films include technically substandard acquisitions that the
model reads as the other class; (ii) borderline films are both harder to
acquire well and harder to label. Either way the flags track measurable
acquisition differences, and the per-image rows
(\texttt{{results/flagged\_property\_rows.csv}}) let any reader inspect
the claim film by film.
"""
    nf = d['n_flagged']; nc = d['n_control']
    head = ("\\section{Flagged-image forensics}\n"
            f"Flagged and control groups ($n={nf}$ flagged, $n={nc}$ controls) "
            "compared on Laplace-variance sharpness, RMS contrast, and Shannon "
            "entropy (scikit-image; tests by pingouin):\n")
    return head + table + verdict

def write(path, text):
    with open(path, 'w') as fh: fh.write(text)
    print('wrote', path, len(text))

if __name__ == '__main__':
    for d in ('malaria', 'pneumonia'):
        write(R(f'paper_{d}/sec_forensics.tex'), sec(d))
