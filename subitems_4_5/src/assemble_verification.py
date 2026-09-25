#!/usr/bin/env python3
"""Benchmark source-verification chapters (Rajaraman / Kermany)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def R(*p): return os.path.join(ROOT, *p)
def J(*p):
    with open(R(*p)) as f: return json.load(f)

pv = J('results/pubmed_verification.json')
kv = J('results/kermany_number_verification.json')
ka = J('results/kermany_abstract_claims.json')
rc = J('results/rajaraman_source_claims.json')

MAL = rf"""
\section{{Source verification of the Rajaraman benchmark}}
\subsection{{Bibliographic verification}}
PubMed record verified: PMID {pv['rajaraman2018']['pmid']},
\emph{{{pv['rajaraman2018']['journal']}}} ({pv['rajaraman2018']['date']}),
title as cited (\texttt{{results/pubmed\_verification.json}}).
\subsection{{Full-text number extraction}}
The numbers this paper compares against were extracted from the PMC full
text (JATS XML) twice, with independent parsers (lxml primary, xmltodict
cross-parse; \texttt{{results/tool\_battery\_4.json}}): cell-level accuracy
0.940, VGG-16 feature extractor 0.945, patient-level 0.959, AlexNet AUC
0.981 --- all present in the source tables, both parsers agreeing.
\subsection{{Claim provenance}}
{len(rc)} verbatim source passages are committed in
\texttt{{results/rajaraman\_source\_claims.json}}, including the acquisition
and prior-work context. Two memory-level misquotes were caught and removed
during the audit: (i) 95.9\% is the \emph{{patient}}-level number, not
cell-level; (ii) there is no VGG-19 result in the paper --- a figure of
``97.4\%'' circulating in secondary summaries does not exist in the source.
The comparison table in this paper uses only the double-extracted numbers.
"""

PNE = rf"""
\section{{Source verification of the Kermany benchmark}}
\subsection{{Bibliographic verification}}
PubMed record verified: PMID {pv['kermany2018']['pmid']},
\emph{{{pv['kermany2018']['journal']}}} ({pv['kermany2018']['date']})
(\texttt{{results/pubmed\_verification.json}}). The abstract's pneumonia
passage is committed verbatim (\texttt{{results/kermany\_abstract\_claims.json}}).
\subsection{{The 92.8\% number, corroborated}}
The Cell article's performance tables are paywalled. The 92.8\% accuracy
figure was therefore corroborated through two independent citing sources
with full text available, {kv['verifier_pmcid']} and PMC11759907; the
verifying passage is committed verbatim in
\texttt{{results/kermany\_number\_verification.json}}. We report the number
as the published reference with this provenance stated, and we never
present it as a same-split comparison.
\subsection{{What the number is and is not}}
Kermany et al.\ fine-tuned Inception-v3 at 299px under transfer learning
on a much larger effective training regime; the 92.8\% is binary
normal-vs-pneumonia accuracy on their test set (the same official split
this paper uses). It is a cross-regime reference --- different model
class, input scale, and training scale --- and our gap to it is addressed
by the transfer study, not by claiming parity.
"""

def write(path, text):
    with open(path, 'w') as f: f.write(text)
    print('wrote', path, len(text))

if __name__ == '__main__':
    write(R('paper_malaria/sec_verification.tex'), MAL)
    write(R('paper_pneumonia/sec_verification.tex'), PNE)
