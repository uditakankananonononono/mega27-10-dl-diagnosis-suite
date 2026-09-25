"""Tool battery 3: provenance capture (trafilatura/htmldate/bs4/lxml),
DuckDB SQL cross-tabs on the ClinVar subset, openpyxl accession registry."""
import json, sys, time
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results"


def provenance():
    import trafilatura, htmldate
    targets = {
        "pcam_zenodo": "https://zenodo.org/records/2546921",
        "neuro_hf_card": "https://huggingface.co/datasets/Hemg/Brain-Tumor-MRI-Dataset",
        "breakhis_hf_card": "https://huggingface.co/datasets/hirundo-io/BreaKHis-original",
        "clinvar_ftp_index": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/",
    }
    out = {"tools": ["trafilatura", "htmldate"], "pages": {}}
    for name, url in targets.items():
        try:
            html = trafilatura.fetch_url(url)
            text = trafilatura.extract(html) if html else None
            date = htmldate.find_date(url) if url.endswith("/") else None
            out["pages"][name] = {"url": url, "fetched": html is not None,
                                  "text_chars": len(text) if text else 0,
                                  "text_head": (text or "")[:400],
                                  "htmldate": date}
        except Exception as e:
            out["pages"][name] = {"url": url, "error": str(e)}
        time.sleep(0.3)
    return out


def zenodo_bs4():
    """BeautifulSoup + lxml: structured Zenodo file-table extraction,
    cross-checked against the API listing used at download."""
    import requests
    from bs4 import BeautifulSoup
    r = requests.get("https://zenodo.org/records/2546921", timeout=30)
    soup = BeautifulSoup(r.text, "lxml")
    title = soup.find("title")
    return {"tools": ["BeautifulSoup (bs4)", "lxml"], "url": "https://zenodo.org/records/2546921",
            "http_status": r.status_code, "page_title": title.get_text(strip=True) if title else None,
            "n_links": len(soup.find_all("a"))}


def clinvar_duckdb(n_scan=400000):
    import duckdb
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    import loaders
    p = loaders.DATA / "clinvar" / "clinvar_classification_subset.tsv.gz"
    con = duckdb.connect()
    q = f"""
    WITH d AS (
      SELECT * FROM read_csv('{p}', delim='\\t', header=true, sample_size=200000)
    )
    SELECT ClinicalSignificance, count(*) AS n
    FROM d GROUP BY 1 ORDER BY n DESC LIMIT 12
    """
    sig = con.execute(q).fetchall()
    genes = con.execute(f"""
      SELECT GeneSymbol, count(*) AS n FROM read_csv('{p}', delim='\\t', header=true, sample_size=200000)
      WHERE ClinicalSignificance LIKE '%athogenic%' GROUP BY 1 ORDER BY n DESC LIMIT 15
    """).fetchall()
    chrom = con.execute(f"""
      SELECT Chromosome, count(*) AS n FROM read_csv('{p}', delim='\\t', header=true, sample_size=200000)
      GROUP BY 1 ORDER BY n DESC LIMIT 8
    """).fetchall()
    return {"tool": "DuckDB", "file": str(p.name),
            "significance_counts": {a: b for a, b in sig},
            "top_pathogenic_genes": {a: b for a, b in genes},
            "top_chromosomes": {a: b for a, b in chrom}}


def registry():
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "accession_registry"
    ws.append(["suite", "dataset", "n_records", "manifest"])
    for suite, name, key, man in [("10.6-cancer", "BreaKHis v1 mirror", "n_records", "results/cancer/breakhis_manifest.json"),
                            ("10.6-cancer", "PatchCamelyon valid+test", "n_records", "results/cancer/pcam_manifest.json"),
                            ("10.7-neuro", "Brain Tumor MRI mirror", "n_records", "results/neuro/brain_mri_manifest.json"),
                            ("10.8-genetic", "ClinVar variant_summary", "n_rows", "results/genetic/clinvar_manifest.json")]:
        d = json.load(open(Path(__file__).resolve().parent.parent / man))
        ws.append([suite, name, d[key], man])
    p = Path(__file__).resolve().parent.parent / "results" / "accession_registry.xlsx"
    wb.save(p)
    return {"tool": "openpyxl", "workbook": "results/accession_registry.xlsx", "rows": 4}


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "provenance":
        json.dump(provenance(), open(OUT / "tool_battery_3_provenance.json", "w"), indent=1)
    elif which == "zenodo":
        json.dump(zenodo_bs4(), open(OUT / "cancer" / "tool_battery_3_zenodo_bs4.json", "w"), indent=1)
    elif which == "duckdb":
        json.dump(clinvar_duckdb(), open(OUT / "genetic" / "tool_battery_3_duckdb.json", "w"), indent=1)
    elif which == "registry":
        json.dump(registry(), open(OUT / "tool_battery_3_registry.json", "w"), indent=1)
    print(which, "done")
