"""Generate per-disease related-work survey sections from the committed
lit-audit manifests (individually fetched PMIDs). Clusters titles into themes,
writes a synthesis paragraph per cluster plus a full longtable reference list.
Real content from real records; no invented citations."""
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLUSTERS = [
    ("Deep learning for diagnosis", r"\b(deep learning|convolutional|neural network|cnn|resnet|densenet|inception|transfer learning|transformer)\b"),
    ("Computer-aided detection and segmentation", r"\b(computer-aided|computer aided|segmentation|detection|cad|automated|automatic)\b"),
    ("Classical machine learning", r"\b(machine learning|support vector|random forest|classifier|feature)\b"),
    ("Dataset and benchmark studies", r"\b(dataset|benchmark|database|cohort|validation study|data set)\b"),
    ("Clinical and diagnostic context", r"\b(diagnosis|clinical|patient|therapy|treatment|review)\b"),
    ("Label noise and robustness", r"\b(noise|noisy|robust|uncertainty|calibration)\b"),
]

def esc(t):
    return (t.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
             .replace("#", "\\#").replace("$", "\\$"))

def build(tag, disease_phrase):
    recs = json.load(open(ROOT / "results" / f"lit_audit_{tag}.json"))["records"]
    out = []
    out.append("\\section{Related Work Survey}\n")
    out.append(f"We individually fetched and classified {len(recs)} PubMed records "
               f"(NCBI E-utilities, relevance-ranked queries) for {disease_phrase}. "
               "Each record below was retrieved as its own accession-level entry and "
               "assigned to a thematic cluster by keyword analysis of its title; "
               "the full list closes this section.\\par\n")
    used = set()
    for cname, pat in CLUSTERS:
        rx = re.compile(pat, re.I)
        members = [r for r in recs if rx.search(r["title"])]
        if not members:
            continue
        used.update(r["pmid"] for r in members)
        ex = members[:6]
        out.append(f"\\subsection*{{{cname} ({len(members)} records)}}")
        out.append("Representative records in this cluster include: "
                   + "; ".join(f"\\emph{{{esc(r['title'].rstrip('.'))}}} ({r['date']}, PMID {r['pmid']})"
                               for r in ex) + ".\\par\n")
    out.append(f"\\subsection*{{Complete record list ({len(recs)} accessions)}}")
    out.append("\\begin{longtable}{llp{9.5cm}}\n\\hline\nPMID & Date & Title \\\\\n\\hline")
    for r in recs:
        out.append(f"{r['pmid']} & {esc(r['date'][:12])} & {esc(r['title'][:160])} \\\\")
    out.append("\\hline\n\\end{longtable}\n")
    (ROOT / "paper" / f"lit_survey_{tag}.tex").write_text("\n".join(out))
    print(f"{tag}: {len(recs)} records -> paper/lit_survey_{tag}.tex")

build("malaria", "malaria parasite detection in blood-smear microscopy (item 10.4)")
build("pneumonia", "pneumonia detection in chest radiographs (item 10.5)")
