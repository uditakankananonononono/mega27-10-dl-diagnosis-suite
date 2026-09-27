"""Recover area fold0 result from completed ckpt (ep7 saved) - eval only, no retraining."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, '.')
from src.common.data import NpyDataset
from src.common.models import GlobalCNNClassifier
from src.common.train import full_metrics

ROOT = Path('.').resolve()
variant, fold = "area", 0
OUT = ROOT / "results" / "malaria" / "resize_diag" / variant
res_p = OUT / f"fold{fold}_result.json"
assert not res_p.exists(), "already done"
ids = np.load(ROOT / "data" / "malaria" / f"malaria48_{variant}_ids.npy")
codes = sorted({s.split("/")[1].split("_IMG")[0] for s in ids})
rng = np.random.default_rng(2026)
perm = rng.permutation(len(codes))
folds = [[] for _ in range(5)]
for i, c in enumerate(perm):
    folds[i % 5].append(codes[c])
code_of = np.array([s.split("/")[1].split("_IMG")[0] for s in ids])
te_idx = np.where(np.isin(code_of, folds[fold]))[0]
y = np.load(ROOT / "data" / "malaria" / f"malaria48_{variant}_y.npy")
st = torch.load(OUT / f"fold{fold}_ckpt.pt", weights_only=False)
assert st["epoch"] == 7 and st.get("batch", 0) == 0, f"ckpt incomplete: ep{st['epoch']} b{st.get('batch')}"
model = GlobalCNNClassifier(3)
model.load_state_dict(st["model"]); model.eval()
ds_ev = NpyDataset(str(ROOT / "data" / "malaria" / f"malaria48_{variant}"), train=False)
probs = []
with torch.no_grad():
    for xb, yb in DataLoader(Subset(ds_ev, te_idx), batch_size=256):
        probs.append(torch.softmax(model(xb), dim=1).numpy())
p = np.concatenate(probs)
m = full_metrics(p, y[te_idx])
tr_clusters = set().union(*[folds[j] for j in range(5) if j != fold])
res = {"fold": fold, "n_test": int(len(te_idx)),
       "n_train": int(len(np.where(np.isin(code_of, list(tr_clusters)))[0])),
       "metrics": m, "hist": st["hist"], "recovered_from_ckpt": True}
json.dump(res, open(res_p, "w"), indent=1)
print("RECOVERED FOLD DONE", fold, m, flush=True)
