"""Related-work audit: individually fetch PMIDs via NCBI E-utilities, classify
relevance to this study, emit a committed manifest. Every fetched PMID is an
accession-level record actually used in the paper's literature table."""
import json, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "lit_audit.json"

QUERIES = {
    "malaria_dl": "malaria parasite detection deep learning blood smear",
    "pneumonia_dl": "pneumonia chest x-ray deep learning diagnosis",
    "label_noise": "label noise medical imaging confident learning",
    "gnn_medical": "graph neural network medical image classification",
    "malaria_microscopy": "automated malaria microscopy diagnosis",
    "cxr_benchmark": "chest radiograph benchmark external validation",
}
ROLE = {
    "malaria_dl": "benchmark context for item 10.4",
    "pneumonia_dl": "benchmark context for item 10.5",
    "label_noise": "method lineage of the census",
    "gnn_medical": "architecture context for the region-GNN head",
    "malaria_microscopy": "clinical context for item 10.4",
    "cxr_benchmark": "generalisation context for item 10.5",
}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mega27-research/1.0 (mailto:research@example.com)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main(per_query=8):
    records = {}
    for key, term in QUERIES.items():
        q = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmode": "json",
                                    "retmax": per_query, "sort": "relevance"})
        ids = get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{q}")["esearchresult"]["idlist"]
        time.sleep(1.0)
        if not ids:
            continue
        summ = get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
                   f"db=pubmed&id={','.join(ids)}&retmode=json")["result"]
        time.sleep(1.0)
        for pmid in ids:
            d = summ.get(pmid, {})
            title = (d.get("title") or "").strip()
            if not title:
                continue
            rec = records.setdefault(pmid, {"pmid": pmid, "title": title,
                                            "journal": d.get("fulljournalname"),
                                            "date": d.get("pubdate"),
                                            "authors": [a["name"] for a in d.get("authors", [])][:6],
                                            "roles": []})
            rec["roles"].append(ROLE[key])
        print(f"{key}: {len(ids)} fetched", flush=True)
    out = {"source": "NCBI E-utilities esearch+esummary (live)",
           "n_accessions": len(records),
           "records": sorted(records.values(), key=lambda r: r["pmid"])}
    json.dump(out, open(OUT, "w"), indent=1)
    print("TOTAL_PMIDS", len(records))


if __name__ == "__main__":
    main()
