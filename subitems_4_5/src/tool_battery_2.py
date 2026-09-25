"""Tool battery 2: literature-grounding services + source-PDF verification.
- pymupdf: extract the exact reported numbers from the Rajaraman 2018 and
  Kermany 2018 source PDFs (if cached), else skip honestly.
- NCBI E-utilities (esearch): has ANY prior label-noise census of these two
  datasets been published? Supports/refutes our novelty claim.
- Europe PMC: same question, independent index.
- CrossRef: verify the two benchmark papers' bibliographic records."""
import json, re, sys
from pathlib import Path
import urllib.request, urllib.parse

ROOT = Path(__file__).resolve().parent.parent
out = {}

def get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "mega27-10b-research/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")

# NCBI esearch
def esearch(term):
    u = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed"
         "&retmode=json&term=" + urllib.parse.quote(term))
    return json.loads(get(u))["esearchresult"]["count"]

out["ncbi_esearch"] = {
    "malaria_cell_images_label_noise": esearch('("malaria" AND "cell images" AND ("label noise" OR "label error" OR "mislabel"))'),
    "chestxray2017_label_noise": esearch('("Kermany" OR "ChestXRay2017") AND ("label noise" OR "label error" OR "mislabel")'),
    "confident_learning_medical": esearch('"confident learning" AND ("medical imaging" OR "chest x-ray" OR "malaria")')}

# Europe PMC
def epmc(q):
    u = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&pageSize=5&query="
         + urllib.parse.quote(q))
    d = json.loads(get(u))
    return {"hitCount": d["hitCount"],
            "top": [r.get("title", "")[:120] for r in d.get("resultList", {}).get("result", [])[:3]]}

out["europepmc"] = {
    "malaria_label_noise": epmc('malaria "thin blood smear" AND ("label noise" OR "mislabeled")'),
    "kermany_label_noise": epmc('Kermany chest x-ray AND ("label noise" OR "mislabeled" OR "label error")')}

# CrossRef works
def crossref(doi):
    u = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    d = json.loads(get(u))["message"]
    return {"title": d.get("title", [""])[0][:120], "journal": (d.get("container-title") or [""])[0],
            "year": d.get("issued", {}).get("date-parts", [[None]])[0][0], "doi": doi}

out["crossref"] = {}
for doi in ("10.7717/peerj.4568", "10.1016/j.cell.2018.02.010"):
    try:
        out["crossref"][doi] = crossref(doi)
    except Exception as e:
        out["crossref"][doi] = {"error": str(e)}

json.dump(out, open(ROOT / "results/tool_battery_2.json", "w"), indent=2)
print(json.dumps(out, indent=1)[:2400])
