"""Domain-shift/modality-diversity battery over BBBC accessions.
Per accession: fetch real images (full zip if <=60MB else example images),
compute a standardized statistics battery, record everything in a manifest.
Each accession is individually fetched and used."""
import io, json, re, sys, time, zipfile
from pathlib import Path

import numpy as np
import urllib.request
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "data" / "bbbc"
OUT = ROOT / "results" / "bbbc_battery.json"
MAX_ZIP = 60_000_000
UA = {"User-Agent": "mega27-research/1.0"}


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def head_size(url):
    try:
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def stats_battery(arr: np.ndarray) -> dict:
    a = arr.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(-1)
    a = a / (a.max() + 1e-9)
    hist, _ = np.histogram(a, bins=32, range=(0, 1), density=True)
    p = hist / (hist.sum() + 1e-12)
    ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return {"height": int(arr.shape[0]), "width": int(arr.shape[1]),
            "channels": int(arr.shape[2]) if arr.ndim == 3 else 1,
            "mean_intensity": float(a.mean()), "std_intensity": float(a.std()),
            "entropy_bits": ent,
            "grad_energy": float(np.mean(np.diff(a, axis=0) ** 2) + np.mean(np.diff(a, axis=1) ** 2))}


def process_accession(acc: str) -> dict:
    html = fetch(f"https://bbbc.broadinstitute.org/{acc}").decode("utf-8", "ignore")
    zips = sorted(set(re.findall(r'https://data\.broadinstitute\.org/bbbc/' + acc + r'/[^"\']+?\.zip', html)))
    examples = sorted(set(re.findall(r'https://data\.broadinstitute\.org/bbbc/' + acc + r'/[^"\']+?\.(?:png|jpg|jpeg|tif)', html)))
    title = (re.search(r"<title>(.*?)</title>", html, re.S) or [None, ""])[1].strip()[:120]
    images = []
    source = None
    for z in zips:
        sz = head_size(z)
        if 0 < sz <= MAX_ZIP:
            blob = fetch(z)
            zf = zipfile.ZipFile(io.BytesIO(blob))
            for n in zf.namelist():
                if n.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    try:
                        images.append(np.asarray(Image.open(io.BytesIO(zf.read(n))).convert("RGB")))
                    except Exception:
                        pass
                    if len(images) >= 12:
                        break
            source = f"{z.split('/')[-1]} ({sz} bytes, {len(images)} images sampled)"
            break
    if not images and examples:
        for u in examples[:6]:
            try:
                images.append(np.asarray(Image.open(io.BytesIO(fetch(u))).convert("RGB")))
            except Exception:
                pass
        source = f"{len(images)} example images"
    if not images:
        return {"accession": acc, "title": title, "status": "no_fetchable_images",
                "zip_urls": len(zips), "example_urls": len(examples)}
    per = [stats_battery(im) for im in images]
    agg = {k: float(np.mean([p[k] for p in per])) for k in
           ("mean_intensity", "std_intensity", "entropy_bits", "grad_energy")}
    agg["accession"] = acc
    agg["title"] = title
    agg["status"] = "used"
    agg["source"] = source
    agg["n_images"] = len(images)
    agg["resolutions"] = sorted({(p["height"], p["width"]) for p in per})[:4]
    return agg


def main():
    sets = (ROOT / "data" / "bbbc_sets.txt").read_text().split()
    results = {}
    for acc in sets:
        try:
            r = process_accession(acc)
        except Exception as e:
            r = {"accession": acc, "status": f"error: {str(e)[:80]}"}
        results[acc] = r
        print(acc, r["status"], flush=True)
        json.dump({"source": "Broad Bioimage Benchmark Collection, per-accession fetch",
                   "results": results}, open(OUT, "w"), indent=1)
        time.sleep(0.5)
    used = sum(1 for r in results.values() if r.get("status") == "used")
    print("USED", used, "of", len(results))


if __name__ == "__main__":
    main()
