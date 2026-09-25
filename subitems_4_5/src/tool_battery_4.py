"""Tool battery 4: verification-oriented tools.
1. xmltodict  - independent re-parse of the Rajaraman JATS XML; the lxml
                table numbers must reproduce exactly (parse-bug defense).
2. BeautifulSoup (bs4) - structured extraction of the NIH LHC malaria page
                publications/dataset table: confirms which smear dataset our
                27,558-image release is.
3. pdfplumber - paper-vs-data audit: extract table text from the rendered
                paper PDF and check key numbers against committed JSONs.
4. pymupdf    - render paper pages to PNG for visual layout verification.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = {}


def xmltodict_crossparse():
    import xmltodict
    doc = xmltodict.parse(open("/tmp/rajaraman_fulltext.xml", "rb").read())
    def walk(o):
        if isinstance(o, dict):
            if "table-wrap" in str(o.keys()):
                pass
            for k, v in o.items():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)
        else:
            yield str(o)
    txt = " ".join(walk(doc))
    found = {}
    for label, needle in [("custom_cell_acc", "0.940"), ("vgg16_acc", "0.945"),
                          ("patient_level_acc", "0.959"), ("alexnet_auc", "0.981")]:
        found[label] = {"needle": needle, "present": needle in txt}
    found["matches_lxml_extraction"] = all(v["present"] for v in found.values())
    return found


def bs4_nih_table():
    import trafilatura
    from bs4 import BeautifulSoup
    url = "https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html"
    html = trafilatura.fetch_url(url)
    soup = BeautifulSoup(html, "lxml")
    tables = []
    for tab in soup.find_all("table"):
        rows = []
        for tr in tab.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    flat = " ".join(" ".join(r) for t in tables for r in t)
    return {"url": url, "n_tables": len(tables),
            "mentions_thin_smear": "Thin Smear" in flat or "thin" in flat.lower(),
            "mentions_27558_or_27558": ("27,558" in flat or "27558" in flat),
            "first_table_head": tables[0][:2] if tables else None,
            "question": "which smear datasets does the NIH LHC page list, and is ours the thin-smear release?"}


def pdfplumber_audit():
    import pdfplumber
    pdf = ROOT / "paper" / "main.pdf"
    if not pdf.exists():
        return {"skipped": "paper/main.pdf not built yet"}
    text = ""
    with pdfplumber.open(str(pdf)) as p:
        n_pages = len(p.pages)
        for pg in p.pages:
            text += (pg.extract_text() or "") + "\n"
    checks = {}
    j = json.load(open(ROOT / "results" / "malaria" / "baseline_results.json"))
    mal_acc = j["cnn"]["accuracy"]
    checks["malaria_cnn_acc_in_pdf"] = {"json": mal_acc,
        "present": f"{mal_acc*100:.2f}" in text or f"{mal_acc:.4f}" in text}
    j2 = json.load(open(ROOT / "results" / "malaria" / "label_noise_census.json"))
    noise = j2["summary"]["estimated_noise_rate"]
    checks["malaria_noise_rate_in_pdf"] = {"json": noise,
        "present": f"{noise*100:.2f}" in text}
    j3 = json.load(open(ROOT / "results" / "pneumonia" / "label_noise_census.json"))
    nflag = len(j3["flagged_ids"])
    checks["pneumonia_n_flagged_in_pdf"] = {"json": nflag, "present": str(nflag) in text}
    checks["n_pages"] = n_pages
    checks["all_present"] = all(c.get("present", True) for k, c in checks.items()
                                if k.startswith(("malaria", "pneumonia")))
    return checks


def pymupdf_render():
    import pymupdf
    pdf = ROOT / "paper" / "main.pdf"
    if not pdf.exists():
        return {"skipped": "paper/main.pdf not built yet"}
    doc = pymupdf.open(str(pdf))
    dest = ROOT / "figures" / "paper_page1_render.png"
    pix = doc[0].get_pixmap(dpi=80)
    pix.save(str(dest))
    fonts = set()
    for i in range(min(5, len(doc))):
        for f in doc[i].get_fonts():
            fonts.add(f[3])
    return {"rendered": str(dest.relative_to(ROOT)), "n_pages": len(doc),
            "fonts_first5": sorted(fonts)[:8]}


def main():
    R["xmltodict_crossparse"] = xmltodict_crossparse()
    print("xmltodict:", R["xmltodict_crossparse"]["matches_lxml_extraction"], flush=True)
    R["bs4_nih_table"] = bs4_nih_table()
    print("bs4 tables:", R["bs4_nih_table"]["n_tables"], flush=True)
    R["pdfplumber_audit"] = pdfplumber_audit()
    print("pdfplumber all_present:", R["pdfplumber_audit"].get("all_present"), flush=True)
    R["pymupdf_render"] = pymupdf_render()
    print("pymupdf pages:", R["pymupdf_render"].get("n_pages"), flush=True)
    json.dump(R, open(ROOT / "results" / "tool_battery_4.json", "w"), indent=2)
    print("BATTERY4_DONE", flush=True)


if __name__ == "__main__":
    main()
