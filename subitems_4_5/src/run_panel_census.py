"""Label-noise atlas across the MedMNIST v2 2D collection (12 accessions).
For each subset: 2-fold OOF probabilities from the CNN core at native 28x28,
confident-joint noise estimate, flagged sample indices + md5 of image bytes.
CPU-feasible: 28x28 images, small epochs, sequential."""
import json, sys, time, hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.models import GlobalCNNClassifier
from src.common.train import train_model, predict_proba, save_json
from src.common.label_noise import noise_summary, flag_label_errors

ROOT = Path(__file__).resolve().parent.parent
MM = ROOT / "data" / "medmnist"
OUT = ROOT / "results" / "panel"


class MedMNISTDataset(Dataset):
    def __init__(self, name, split="train", train=False, seed=0, max_n=None):
        d = np.load(MM / f"{name}.npz")
        self.x = d[f"{split}_images"]
        self.y = d[f"{split}_labels"].ravel().astype(int)
        if max_n and len(self.y) > max_n:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(self.y), max_n, replace=False))
            self.x, self.y = self.x[keep], self.y[keep]
        self.train = train
        self.seed = seed
        self.n_classes = int(self.y.max()) + 1
        self.in_ch = 1 if self.x.ndim == 3 else self.x.shape[-1]

    def __len__(self):
        return len(self.y)

    def sample_id(self, i):
        h = hashlib.md5(self.x[i].tobytes()).hexdigest()[:12]
        return f"idx{i}#md5:{h}#label:{self.y[i]}"

    def __getitem__(self, i):
        arr = self.x[i].astype(np.float32) / 255.0
        if self.train:
            rng = np.random.default_rng(self.seed * 1_000_003 + i)
            if rng.random() < 0.5:
                arr = arr[:, ::-1]
            if rng.random() < 0.5:
                arr = arr[::-1, :]
            arr = arr.copy()
        if arr.ndim == 2:
            arr = arr[None]
        else:
            arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(arr)), int(self.y[i])


def census_one(name, epochs=3, folds=2, seed=0, max_train=5000):
    t0 = time.time()
    ds_eval = MedMNISTDataset(name, "train", train=False, max_n=max_train)
    n, K = len(ds_eval), ds_eval.n_classes
    probs = np.zeros((n, K))
    fold_of = np.arange(n) % folds
    rng = np.random.default_rng(seed)
    rng.shuffle(fold_of)
    for f in range(folds):
        tr = np.where(fold_of != f)[0]; va = np.where(fold_of == f)[0]
        tr_ds = Subset(MedMNISTDataset(name, "train", train=True, seed=seed + f,
                                       max_n=max_train), tr)
        va_ds = Subset(ds_eval, va)
        model = GlobalCNNClassifier(ds_eval.in_ch, n_classes=K)
        import math
        epochs_eff = max(epochs, math.ceil(250 * 128 / max(len(tr), 1)))
        model, _, _ = train_model(model, tr_ds, va_ds, epochs=epochs_eff, batch=128,
                                  seed=seed + f, patience=20)
        p, _ = predict_proba(model, va_ds)
        probs[va] = p
    oof_acc = float((probs.argmax(1) == ds_eval.y).mean())
    majority = float(np.bincount(ds_eval.y).max() / len(ds_eval.y))
    admissible = oof_acc >= max(0.60, majority + 0.05)
    summary = noise_summary(probs, ds_eval.y)
    flags = flag_label_errors(probs, ds_eval.y)
    flagged = [ds_eval.sample_id(int(i)) for i in np.where(flags)[0]]
    out = {"dataset": name, "n": n, "n_classes": K, "epochs": epochs,
           "folds": folds, "max_train": max_train,
           "oof_accuracy": oof_acc, "majority_rate": majority,
           "admissible": admissible,
           "admissibility_rule": "oof_acc >= max(0.60, majority + 0.05); undertrained OOF models make confident-learning estimates invalid, so inadmissible datasets report the gate failure instead of a noise rate",
           "summary": summary if admissible else None,
           "flagged_ids": flagged if admissible else [],
           "secs": round(time.time() - t0, 1)}
    save_json(out, OUT / f"{name}_census.json")
    np.save(OUT / f"{name}_oof_probs.npy", probs)
    tag = f"noise={summary['estimated_noise_rate']:.4f} flagged={len(flagged)}" if admissible else "INADMISSIBLE (gate)"
    print(f"[{name}] K={K} n={n} oof_acc={out['oof_accuracy']:.3f} {tag} ({out['secs']}s)", flush=True)
    return out


NAMES = ["pneumoniamnist", "breastmnist", "retinamnist", "dermamnist", "octmnist",
         "bloodmnist", "organamnist", "organcmnist", "organsmnist", "pathmnist",
         "chestmnist", "tissuemnist"]

if __name__ == "__main__":
    which = sys.argv[1:] or NAMES
    # resume: keep previously completed entries, only compute missing/failed
    prev = OUT / "atlas_summary.json"
    atlas = json.load(open(prev)) if prev.exists() else {}
    for name in list(atlas):
        if name not in which or "error" in (atlas[name] or {}):
            del atlas[name]
    for name in which:
        if name in atlas:
            print(f"[{name}] resume: already complete, skipping", flush=True)
            continue
        try:
            r = census_one(name)
            atlas[name] = {"noise_rate": (r["summary"] or {}).get("estimated_noise_rate"),
                           "flagged": len(r["flagged_ids"]), "n": r["n"], "K": r["n_classes"],
                           "oof_accuracy": r["oof_accuracy"], "admissible": r["admissible"]}
        except Exception as e:
            print(f"[{name}] FAILED: {e}", flush=True)
            atlas[name] = {"error": str(e)}
        save_json(atlas, OUT / "atlas_summary.json")
    print("PANEL_DONE", flush=True)
