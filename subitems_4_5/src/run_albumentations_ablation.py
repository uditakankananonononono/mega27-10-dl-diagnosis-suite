"""Albumentations augmentation ablation on the malaria CNN.
Question: does a richer augmentation policy (shift/scale/rotate +
brightness/contrast) beat the committed flips+rot90 policy?
Arms: no-aug | flips+rot90 (current) | albumentations pipeline.
3 epochs each; committed: results/malaria/augmentation_ablation.json
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.data import NpyDataset
from src.common.models import GlobalCNNClassifier
from src.common.train import train_model, predict_proba
from src.common.metrics import full_metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "malaria"


class AlbDataset(Dataset):
    def __init__(self, base, idxs, transform):
        self.base, self.idxs, self.transform = base, list(map(int, idxs)), transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        j = self.idxs[i]
        arr = np.asarray(self.base.x[j])              # (C,H,W) uint8
        img = np.moveaxis(arr, 0, -1)                  # HWC for albumentations
        img = self.transform(image=img)["image"]
        img = np.moveaxis(img, -1, 0).astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(img)), int(self.base.y[j])


class NoAug(Dataset):
    def __init__(self, base, idxs):
        self.base, self.idxs = base, list(map(int, idxs))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        j = self.idxs[i]
        arr = np.asarray(self.base.x[j], dtype=np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(arr)), int(self.base.y[j])


def main(epochs=3):
    import albumentations as A
    raw = NpyDataset(str(ROOT / "data" / "malaria" / "malaria48"), train=False)
    ev = NpyDataset(str(ROOT / "data" / "malaria" / "malaria48"), train=False)
    split = json.load(open(OUT / "split.json"))
    tr_i, va_i = np.array(split["train"]), np.array(split["val"])
    alb_tf = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=30,
                           border_mode=0, p=0.8),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
    ])
    arms = {
        "no_aug": NoAug(raw, tr_i),
        "flips_rot90_current": Subset(NpyDataset(str(ROOT / "data" / "malaria" / "malaria48"),
                                                 train=True, seed=0), tr_i),
        "albumentations": AlbDataset(raw, tr_i, alb_tf),
    }
    out = {}
    for name, tr_ds in arms.items():
        torch.manual_seed(0)
        model = GlobalCNNClassifier(3)
        t0 = time.time()
        model, hist, best = train_model(model, tr_ds, Subset(ev, va_i),
                                        epochs=epochs, batch=64, seed=0,
                                        patience=2, threads=1)
        probs, y = predict_proba(model, Subset(ev, va_i))
        from sklearn.metrics import roc_auc_score, accuracy_score
        out[name] = {"val_auc": float(roc_auc_score(y, probs)),
                     "val_acc": float(accuracy_score(y, probs > 0.5)),
                     "train_secs": round(time.time() - t0, 1)}
        print(name, out[name], flush=True)
    json.dump(out, open(OUT / "augmentation_ablation.json", "w"), indent=2)
    print("ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
