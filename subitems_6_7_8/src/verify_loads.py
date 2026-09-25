"""Real-data smoke verification: decode N records from each verified dataset
and record measured properties as committed JSON evidence."""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders

OUT = Path(__file__).resolve().parent.parent / "results"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50

def smoke(name, it, keys):
    t0 = time.time()
    shapes, labels, ids = [], [], []
    for i, r in enumerate(it):
        if i >= N:
            break
        ids.append(r[keys["id"]])
        labels.append(int(r["label"]))
        if "image" in r:
            shapes.append(list(r["image"].shape))
    d = {"dataset": name, "n_decoded": len(ids), "wall_seconds": round(time.time() - t0, 2),
         "first_ids": ids[:5], "label_counts": {str(v): labels.count(v) for v in sorted(set(labels))}}
    if shapes:
        d["unique_shapes"] = sorted({tuple(s) for s in shapes}, key=str)[:10]
        d["n_unique_shapes"] = len({tuple(s) for s in shapes})
    return d

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "breakhis":
        d = smoke("breakhis", loaders.iter_breakhis(), {"id": "image_id"})
        json.dump(d, open(OUT / "cancer" / "breakhis_loader_smoke.json", "w"), indent=1)
    elif which == "pcam":
        for split in ("valid", "test"):
            d = smoke(f"pcam_{split}", loaders.iter_pcam(split), {"id": "patch_id"})
            json.dump(d, open(OUT / "cancer" / f"pcam_{split}_loader_smoke.json", "w"), indent=1)
    elif which == "neuro":
        d = smoke("neuro", loaders.iter_neuro(), {"id": "record_id"})
        json.dump(d, open(OUT / "neuro" / "brain_mri_loader_smoke.json", "w"), indent=1)
    elif which == "clinvar":
        it = loaders.iter_clinvar_subset()
        t0 = time.time(); n = 0; pos = 0; genes = set()
        for r in it:
            n += 1; pos += r["label"]; genes.add(r["gene"])
            if n >= 200000:
                break
        d = {"dataset": "clinvar_subset", "n_streamed": n, "n_pathogenic_side": pos,
             "n_benign_side": n - pos, "n_unique_genes": len(genes),
             "wall_seconds": round(time.time() - t0, 2)}
        json.dump(d, open(OUT / "genetic" / "clinvar_loader_smoke.json", "w"), indent=1)
    print(which, "done")
