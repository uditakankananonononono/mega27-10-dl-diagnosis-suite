#!/usr/bin/env python3
"""Publication-year trend tables + analysis for both surveys."""
import json, collections, re, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for d in ('malaria', 'pneumonia'):
    lit = json.load(open(os.path.join(ROOT, f'results/lit_audit_{d}.json')))
    recs = lit.get('records', lit if isinstance(lit, list) else [])
    years = collections.Counter()
    for r in recs:
        m = re.search(r'(19|20)\d{2}', str(r.get('date', '') or r.get('year', '')))
        if m: years[m.group(0)] += 1
    ys = sorted(years)
    rows = "\n".join(f"{y} & {years[y]} \\\\" for y in ys)
    peak = max(ys, key=lambda y: years[y])
    recent = sum(years[y] for y in ys if int(y) >= 2022)
    if d == 'malaria':
        note = ("The malaria literature is old (first record 1999), "
                "accelerated sharply after the 2018 benchmark reference, and "
                f"peaked in {peak} with {years[peak]} screened records; "
                f"{recent} of 197 records are from 2022 onward --- the "
                "collection is still actively used, which makes the absence "
                "of a label-noise audit a live gap, not a historical one.")
    else:
        note = ("The pneumonia literature is entirely post-2018 (the "
                f"collection's publication year), peaked in {peak} with "
                f"{years[peak]} screened records, and remains active; "
                f"{recent} of 150 records are from 2022 onward.")
    tex = (f"\\subsection*{{Publication-year trend}}\n"
           f"\\begin{{tabular}}{{lr}}\n\\hline\nYear & Records \\\\\n\\hline\n"
           f"{rows}\n\\hline\n\\end{{tabular}}\n\n{note}\\par\n")
    for side in ('paper', f'paper_{d}'):
        p = os.path.join(ROOT, side, f'lit_survey_{d}.tex')
        if not os.path.exists(p): continue
        s = open(p).read()
        j = s.find('\\subsection*{Complete record list')
        s = s[:j] + tex + '\n' + s[j:]
        open(p, 'w').write(s)
        print('trend table added:', p)
