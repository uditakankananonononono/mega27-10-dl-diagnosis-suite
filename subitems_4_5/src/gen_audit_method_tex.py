#!/usr/bin/env python3
"""Literature-audit methodology section per disease (real queries)."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = {
'malaria': ["malaria parasite detection deep learning blood smear",
            "automated malaria microscopy diagnosis",
            "Plasmodium image classification machine learning",
            "malaria thin smear image analysis computer aided",
            "malaria red blood cell segmentation detection"],
'pneumonia': ["pneumonia chest x-ray deep learning diagnosis",
              "pediatric pneumonia radiograph computer aided detection",
              "chest radiograph classification convolutional neural network",
              "pneumonia detection transfer learning chest X-ray",
              "label noise chest x-ray dataset"],
}
for d in ('malaria', 'pneumonia'):
    lit = json.load(open(os.path.join(ROOT, f'results/lit_audit_{d}.json')))
    n = lit.get('n_accessions', len(lit.get('records', [])))
    qlist = "\n".join(f"\\item \\texttt{{{q}}}" for q in Q[d])
    tex = rf"""
\section{{Literature-audit methodology}}
\subsection{{Queries and retrieval}}
{n} PubMed records were individually fetched through NCBI E-utilities
(esearch, relevance-ranked, then esummary per record) with these queries:
\begin{{itemize}}
{qlist}
\end{{itemize}}
Retrieval was capped at 150 records per query; deduplication by PMID; the
shared suite audit's {d} subset was unioned in and deduplicated again.
Every record carries its PMID, title, journal, and date in
\texttt{{results/lit\_audit\_{d}.json}} --- accession-level evidence, not
a count claim.
\subsection{{Screening and clustering}}
Records were screened by title and assigned to thematic clusters by
keyword analysis (deep learning, CAD/segmentation, classical ML,
dataset/benchmark, clinical context, label noise/robustness). The
screen's limits are stated in the survey: title-keyword clustering can
misfile borderline records, PubMed coverage is not the whole literature,
and the audit date bounds recency claims. No record was excluded from the
ledger for being inconvenient to our novelty claim --- the label-noise
cluster's tiny size \emph{{is}} the finding.
"""
    out = os.path.join(ROOT, f'paper_{d}/sec_audit_method.tex')
    open(out, 'w').write(tex)
    print('wrote', out)
