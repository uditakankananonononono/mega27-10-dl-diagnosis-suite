"""Tool battery 3: chained, question-driven uses of six research tools.

1. imageio  - independent third decoder cross-check: do source image bytes
              decode to the same pixels our pretensor pipeline stored?
2. duckdb   - SQL cross-tabs over census + per-image property tables:
              how do flags distribute across classes/datasets?
3. formulaic- adjusted logistic model: which image property independently
              predicts a census flag (vs the unadjusted Mann-Whitney)?
4. plotly   - interactive property-scatter artifact (HTML) for review.
5. openpyxl - flagged-image registry workbook (one sheet per disease).
6. trafilatura - provenance capture: clean text/metadata of the Kermany
              dataset landing page (title, license, version).
"""
import io, json, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.data import NpyDataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
R = {}


def imageio_decode_check():
    """imageio: independent decode of source files vs stored pretensor arrays."""
    import imageio.v2 as imageio
    checks = []
    # malaria: PNGs under cell_images/<class>/<file>.png ; ids are class/file
    ds = NpyDataset(str(ROOT / "data" / "malaria" / "malaria48"), train=False)
    rng = np.random.default_rng(1)
    base = ROOT / "data" / "malaria" / "cell_images"
    ok = fail = 0
    for i in rng.choice(len(ds), 60, replace=False):
        sid = str(ds.ids[i])
        p = base / sid
        if not p.exists():
            continue
        src = imageio.imread(p)
        if src.ndim == 2:
            src = np.stack([src] * 3, -1)
        from PIL import Image
        src = np.asarray(Image.fromarray(src.astype(np.uint8)).convert("RGB").resize(
            (48, 48), Image.BILINEAR))
        # replicate pretensor exactly (RGB convert + BILINEAR); imageio supplies the decode
        d = np.abs(src.astype(int) - np.moveaxis(np.asarray(ds.x[i]), 0, -1).astype(int)).max()
        checks.append(int(d)); ok += d <= 2; fail += d > 2
    mal = {"sampled": len(checks), "max_abs_pixel_diff": max(checks) if checks else None,
           "n_within_tol_2": int(ok), "n_beyond_tol": int(fail)}
    # pneumonia: JPEGs inside ChestXRay2017.zip
    dsp = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_train"), train=False)
    zf = zipfile.ZipFile(ROOT / "data" / "pneumonia" / "ChestXRay2017.zip")
    names = set(zf.namelist())
    checks = []
    for i in rng.choice(len(dsp), 40, replace=False):
        sid = str(dsp.ids[i])  # e.g. NORMAL/IM-0001-0001.jpeg
        cand = f"chest_xray/train/{sid}"
        if cand not in names:
            continue
        src = imageio.imread(io.BytesIO(zf.read(cand)))
        if src.ndim == 3:
            src = np.asarray(Image.fromarray(src.astype(np.uint8)).convert("L"))
        from PIL import Image
        src = np.asarray(Image.fromarray(src.astype(np.uint8)).convert("L").resize(
            (128, 128), Image.BILINEAR))
        stored = np.asarray(dsp.x[i])
        stored = stored[0] if stored.ndim == 3 else stored
        checks.append(int(np.abs(src.astype(int) - stored.astype(int)).max()))
    pne = {"sampled": len(checks), "max_abs_pixel_diff": max(checks) if checks else None}
    return {"malaria": mal, "pneumonia": pne,
            "question": "independent imageio decode matches pretensor-stored arrays?"}


def property_rows(disease, prefix, n_control=500):
    """Per-image property rows (same deterministic control sample as the
    flagged-properties analysis, seed 0)."""
    from skimage.filters import laplace
    from skimage.measure import shannon_entropy
    ds = NpyDataset(prefix, train=False)
    census = json.load(open(OUT / disease / "label_noise_census.json"))
    split = json.load(open(OUT / disease / "split.json"))
    tr = np.array(split["train"])
    flagged_ids = set(census["flagged_ids"])
    ids = np.array([str(i) for i in ds.ids])
    flag_mask = np.array([i in flagged_ids for i in ids[tr]])
    flag_idx, clean_idx = tr[flag_mask], tr[~flag_mask]
    rng = np.random.default_rng(0)
    clean_sub = rng.choice(clean_idx, min(n_control, len(clean_idx)), replace=False)
    rows = []
    for lab, idxs in (("flagged", flag_idx), ("control", clean_sub)):
        for i in idxs:
            arr = np.asarray(ds.x[i], dtype=np.float32)
            g = arr.mean(0) if arr.ndim == 3 else arr
            g = g / 255.0
            rows.append({"disease": disease, "sample_id": str(ds.ids[i]),
                         "true_label": int(ds.y[i]), "group": lab,
                         "sharpness": float(laplace(g).var()),
                         "contrast": float(g.std()),
                         "entropy": float(shannon_entropy((g * 255).astype(np.uint8)))})
    return pd.DataFrame(rows)


def duckdb_crosstab(df, census_m, census_p):
    """duckdb: SQL aggregation over the property table + census summaries."""
    import duckdb
    con = duckdb.connect()
    con.register("props", df)
    q1 = con.execute("""
        SELECT disease, "group", count(*) AS n,
               round(median(sharpness), 6) AS med_sharp,
               round(median(contrast), 6) AS med_contrast,
               round(median(entropy), 4) AS med_entropy
        FROM props GROUP BY disease, "group" ORDER BY disease, "group"
    """).df()
    q2 = con.execute("""
        SELECT disease, true_label, sum(case when "group"='flagged' then 1 else 0 end) AS flagged,
               sum(case when "group"='control' then 1 else 0 end) AS control
        FROM props GROUP BY disease, true_label ORDER BY disease, true_label
    """).df()
    out = {"by_group": q1.to_dict("records"), "by_class": q2.to_dict("records")}
    # class balance in flags vs dataset class balance
    out["flag_rate_malaria"] = census_m["summary"]["estimated_noise_rate"]
    out["flag_rate_pneumonia"] = census_p["summary"]["estimated_noise_rate"]
    return out


def formulaic_logit(df):
    """formulaic: adjusted logistic model flagged ~ properties, per disease."""
    import formulaic
    out = {}
    for disease in ("malaria", "pneumonia"):
        d = df[df.disease == disease].copy()
        d["y"] = (d["group"] == "flagged").astype(int)
        # standardize for comparable odds ratios
        for c in ("sharpness", "contrast", "entropy"):
            d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
        mm = formulaic.model_matrix("y ~ sharpness_z + contrast_z + entropy_z", d)
        import statsmodels.api as sm
        fit = sm.Logit(mm.lhs.values.ravel(), mm.rhs).fit(disp=0)
        out[disease] = {
            "n": int(len(d)),
            "odds_ratios_per_sd": {k: round(float(np.exp(fit.params[k])), 3)
                                   for k in fit.params.index if k != "Intercept"},
            "pvalues": {k: float(fit.pvalues[k]) for k in fit.params.index if k != "Intercept"},
            "pseudo_r2": round(float(fit.prsquared), 4),
        }
    return out


def plotly_scatter(df):
    """plotly: interactive sharpness x contrast scatter, colored by flag."""
    import plotly.express as px
    fig = px.scatter(df, x="contrast", y="sharpness", color="group",
                     facet_col="disease", opacity=0.45,
                     hover_data=["sample_id"],
                     title="Census-flagged vs control images: acquisition properties")
    dest = ROOT / "figures" / "flagged_properties_interactive.html"
    fig.write_html(str(dest))
    return {"artifact": str(dest.relative_to(ROOT)), "n_points": int(len(df))}


def openpyxl_registry(df):
    """openpyxl: flagged-image registry workbook."""
    from openpyxl import Workbook
    wb = Workbook(); wb.remove(wb.active)
    for disease in ("malaria", "pneumonia"):
        ws = wb.create_sheet(disease)
        sub = df[df.disease == disease]
        ws.append(["sample_id", "true_label", "group", "sharpness", "contrast", "entropy"])
        for _, r in sub.iterrows():
            ws.append([r.sample_id, int(r.true_label), r.group,
                       round(r.sharpness, 6), round(r.contrast, 6), round(r.entropy, 4)])
    dest = ROOT / "results" / "flagged_image_registry.xlsx"
    wb.save(dest)
    return {"artifact": str(dest.relative_to(ROOT))}


def trafilatura_provenance():
    """trafilatura: provenance capture of the Kermany dataset landing page."""
    import trafilatura
    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC5907772/"
    html = trafilatura.fetch_url(url)
    if not html:
        return {"url": url, "fetched": False}
    meta = trafilatura.extract_metadata(html)
    text = trafilatura.extract(html) or ""
    return {"url": url, "fetched": True,
            "title": getattr(meta, "title", None),
            "text_head": " ".join(text.split())[:400]}


def main():
    print("[1/6] imageio decode cross-check", flush=True)
    R["imageio_decode_check"] = imageio_decode_check()
    print("[2/6] per-image property rows", flush=True)
    df = pd.concat([property_rows("malaria", str(ROOT / "data" / "malaria" / "malaria48")),
                    property_rows("pneumonia", str(ROOT / "data" / "pneumonia" / "cxr_train"))])
    df.to_csv(OUT / "flagged_property_rows.csv", index=False)
    census_m = json.load(open(OUT / "malaria" / "label_noise_census.json"))
    census_p = json.load(open(OUT / "pneumonia" / "label_noise_census.json"))
    print("[3/6] duckdb cross-tabs", flush=True)
    R["duckdb_crosstabs"] = duckdb_crosstab(df, census_m, census_p)
    print("[4/6] formulaic logistic model", flush=True)
    R["formulaic_logit"] = formulaic_logit(df)
    print("[5/6] plotly + openpyxl artifacts", flush=True)
    R["plotly_scatter"] = plotly_scatter(df)
    R["openpyxl_registry"] = openpyxl_registry(df)
    print("[6/6] trafilatura provenance", flush=True)
    R["trafilatura_provenance"] = trafilatura_provenance()
    json.dump(R, open(OUT / "tool_battery_3.json", "w"), indent=2)
    print(json.dumps(R, indent=1)[:3000])
    print("BATTERY3_DONE", flush=True)


if __name__ == "__main__":
    main()
