"""Single-trial worker: trains one (lr) config on eighth-train/quarter-val,
prints AUC as JSON. Fresh process per trial = memory isolation."""
import json, sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common.data import NpyDataset
from src.common.models import GlobalCNNClassifier
from src.common.train import train_model, predict_proba

def main(lr):
    ds = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_train"), train=True, seed=0)
    ev = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_train"), train=False)
    split = json.load(open(ROOT / "results" / "pneumonia" / "split.json"))
    tr_i = np.array(split["train"])[::8]
    va_i = np.array(split["val"])[::4]
    torch.manual_seed(0)
    model = GlobalCNNClassifier(1)
    model, hist, best = train_model(model, Subset(ds, tr_i), Subset(ev, va_i),
                                    epochs=1, batch=32, lr=lr, seed=0, patience=2)
    probs, y = predict_proba(model, Subset(ev, va_i))
    print("AUC_JSON " + json.dumps({"auc": float(roc_auc_score(y, probs[:, 1]))}), flush=True)

if __name__ == "__main__":
    main(float(sys.argv[1]))
