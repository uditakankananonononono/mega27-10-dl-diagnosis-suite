"""Albumentations ablation for the pneumonia CNN (item 10.5):
current flips policy vs albumentations shift/brightness pipeline,
2 epochs each on a quarter-subsample (speed directive). Committed JSON."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common.data import NpyDataset
from src.common.models import GlobalCNNClassifier
from src.common.train import train_model, predict_proba

OUT = ROOT / "results" / "pneumonia"

class AlbDataset(Dataset):
    def __init__(self, base, idxs, transform):
        self.base, self.idxs, self.transform = base, list(idxs), transform
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, i):
        x, y = self.base[self.idxs[i]]
        arr = (x.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        arr = self.transform(image=arr)["image"]
        t = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
        return t, y

def main(epochs=2):
    import albumentations as A
    base = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_train"), train=True, seed=0)
    ev = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_train"), train=False)
    split = json.load(open(OUT / "split.json"))
    tr_i = np.array(split["train"])[::4]
    va_i = np.array(split["val"])
    arms = {
        "flips_current": A.Compose([A.HorizontalFlip(p=0.5)]),
        "albumentations_richer": A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=8, p=0.7),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
        ]),
    }
    res = {}
    for name, tr in arms.items():
        torch.manual_seed(0)
        model = GlobalCNNClassifier(1)
        t0 = time.time()
        model, hist, best = train_model(model, AlbDataset(base, tr_i, tr),
                                        Subset(ev, va_i), epochs=epochs,
                                        batch=32, lr=5e-4, seed=0)
        probs, y = predict_proba(model, Subset(ev, va_i))
        res[name] = {"val_auc": round(float(roc_auc_score(y, probs[:, 1])), 4),
                     "secs": round(time.time() - t0, 1)}
        print(f"[{name}] val_auc={res[name]['val_auc']}", flush=True)
    res["note"] = "quarter-subsample, 2 epochs/arm (speed directive)"
    json.dump(res, open(OUT / "augmentation_ablation.json", "w"), indent=1)
    print("ABLATION_PNEU_DONE", flush=True)

if __name__ == "__main__":
    main()
