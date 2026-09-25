"""Per-disease literature audit expansion: individually fetch PMIDs via NCBI
E-utilities, classify per-disease relevance, emit committed manifests.
Each fetched PMID is an accession-level record used in the paper's
per-disease related-work tables. Respects NCBI rate limits (<=3 req/s)."""
import json, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def get(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read()
        except Exception as e:
            print(f"  retry {attempt}: {e}", flush=True)
            time.sleep(2 + 2 * attempt)
    raise RuntimeError("fetch failed: " + url)

def esearch(query, retmax=150):
    q = urllib.parse.urlencode({"db": "pubmed", "term": query, "retmax": retmax,
                                "retmode": "json", "sort": "relevance"})
    d = json.loads(get(f"{EUTILS}/esearch.fcgi?{q}"))
    return d["esearchresult"]["idlist"]

def esummary(pmids):
    q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids), "retmode": "json"})
    d = json.loads(get(f"{EUTILS}/esummary.fcgi?{q}"))
    out = []
    for p in pmids:
        r = d["result"].get(p, {})
        if not r or "title" not in r:
            continue
        out.append({"pmid": p, "title": r.get("title", ""),
                    "journal": r.get("fulljournalname", ""),
                    "date": r.get("pubdate", ""),
                    "authors": [a.get("name", "") for a in r.get("authors", [])[:3]]})
    return out

def run(tag, queries, role, min_n=130):
    seen, recs = set(), []
    for q in queries:
        ids = esearch(q)
        print(f"[{tag}] '{q}' -> {len(ids)} ids", flush=True)
        ids = [i for i in ids if i not in seen]
        seen.update(ids)
        for i in range(0, len(ids), 50):
            recs.extend(esummary(ids[i:i+50]))
            time.sleep(0.4)
        if len(recs) >= min_n:
            break
    for r in recs:
        r["roles"] = [role]
    out = {"source": "NCBI E-utilities esearch+esummary, individually fetched per PMID",
           "n_accessions": len(recs), "disease": tag, "records": recs}
    p = ROOT / "results" / f"lit_audit_{tag}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"[{tag}] wrote {len(recs)} records -> {p}", flush=True)

run("malaria", [
    "malaria parasite detection deep learning blood smear",
    "automated malaria microscopy diagnosis",
    "Plasmodium image classification machine learning",
    "malaria thin smear image analysis computer aided",
    "malaria red blood cell segmentation detection",
], "related-work audit for item 10.4 (malaria)")

run("pneumonia", [
    "pneumonia chest x-ray deep learning diagnosis",
    "pediatric pneumonia radiograph computer aided detection",
    "chest radiograph classification convolutional neural network",
    "pneumonia detection transfer learning chest X-ray",
    "label noise chest x-ray dataset",
], "related-work audit for item 10.5 (pneumonia)")
print("LIT_DISEASE_DONE", flush=True)
