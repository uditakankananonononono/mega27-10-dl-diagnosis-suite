"""Tool battery 5:
1. htmldate  - extract the publication/modified date of the NIH LHC malaria
               datasheet page: date evidence for dataset provenance.
2. reportlab - generate the one-page census review card PDF (deliverable
               summary of both label-noise censuses for reviewers).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = {}


def htmldate_provenance():
    import trafilatura, htmldate
    url = "https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-datasheet.html"
    html = trafilatura.fetch_url(url)
    d = htmldate.find_date(html, original_date=True) if html else None
    m = htmldate.find_date(html, original_date=False) if html else None
    return {"url": url, "original_date": d, "latest_date": m,
            "question": "when was the NIH malaria dataset record published/updated?"}


def reportlab_card():
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    out = ROOT / "results" / "census_review_card.pdf"
    c = canvas.Canvas(str(out), pagesize=letter)
    w, h = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.9 * inch, h - 1.0 * inch, "Label-noise census review card")
    c.setFont("Helvetica", 10)
    y = h - 1.4 * inch
    for disease in ("malaria", "pneumonia"):
        j = json.load(open(ROOT / "results" / disease / "label_noise_census.json"))
        x = json.load(open(ROOT / "results" / disease / "census_crosscheck.json"))
        s = j["summary"]
        pce = s["per_class_error"]
        lines = [
            f"{disease.upper()}  (n={s['n']} train images scored)",
            f"  estimated noise rate: {s['estimated_noise_rate']*100:.2f}%  "
            f"(est. label errors: {s['estimated_label_errors']})",
            f"  asymmetry: {pce}",
            f"  flagged IDs published: {len(j['flagged_ids'])}  "
            f"(cleanlab independent agreement, Jaccard {x['jaccard']:.3f})",
            "",
        ]
        for ln in lines:
            c.drawString(0.9 * inch, y, ln); y -= 0.22 * inch
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(0.9 * inch, y - 0.1 * inch,
                 "All numbers from committed results/*/label_noise_census.json and census_crosscheck.json.")
    c.save()
    return {"artifact": "results/census_review_card.pdf"}


def main():
    R["htmldate_provenance"] = htmldate_provenance()
    print("htmldate:", R["htmldate_provenance"], flush=True)
    R["reportlab_card"] = reportlab_card()
    print("reportlab:", R["reportlab_card"], flush=True)
    json.dump(R, open(ROOT / "results" / "tool_battery_5.json", "w"), indent=2)
    print("BATTERY5_DONE", flush=True)


if __name__ == "__main__":
    main()
