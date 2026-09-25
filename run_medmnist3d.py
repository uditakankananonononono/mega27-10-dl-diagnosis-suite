"""MedMNIST 3D arm: 6 published 3D medical-image benchmark subsets, real 3D CNNs.

Same protocol as the 2D arm (run_medmnist.py): small CNN, 3 epochs, Adam 1e-3,
test-set AUC (macro OvR for multi-class). Datasets from MedMNIST v2 Zenodo
record 10519652. Resume-safe via results/medmnist3d.json.
"""
import json, os, socket, sys, time, warnings, urllib.request
socket.setdefaulttimeout(120)
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
import torch; torch.set_num_threads(2)
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from medmnist import INFO

SUBSETS = ["organmnist3d", "nodulemnist3d", "adrenalmnist3d",
           "fracturemnist3d", "vesselmnist3d", "synapsemnist3d"]
URL = "https://zenodo.org/records/10519652/files/{}.npz?download=1"

class SmallCNN3D(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, n_classes))
    def forward(self, x):
        return self.f(x)

def run_subset(name, epochs=3, seed=0):
    info = INFO[name]
    z = np.load(f"data_cache/medmnist/{name}.npz")
    tr_imgs = z["train_images"]; tr_labels = z["train_labels"]
    te_imgs = z["test_images"]; te_labels = z["test_labels"]
    n_classes = len(info["label"]); task = info["task"]
    torch.manual_seed(seed)
    model = SmallCNN3D(n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ytr = torch.tensor(tr_labels.squeeze(), dtype=torch.long)
    bs, ntr = 32, len(tr_imgs)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(ntr)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            xb = torch.tensor(tr_imgs[idx.numpy()], dtype=torch.float32).unsqueeze(1) / 255.0
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    yte = te_labels.squeeze()
    with torch.no_grad():
        logits = torch.cat([model(torch.tensor(te_imgs[i:i + 64], dtype=torch.float32).unsqueeze(1) / 255.0)
                            for i in range(0, len(te_imgs), 64)])
    prob = torch.softmax(logits, dim=1).numpy()
    if n_classes == 2:
        auc = roc_auc_score(yte, prob[:, 1])
    else:
        yoh = np.eye(n_classes)[yte]
        auc = roc_auc_score(yoh, prob, average="macro", multi_class="ovr")
    return {"subset": name, "task": task, "n_classes": n_classes,
            "n_train": len(tr_imgs), "n_test": len(te_imgs), "auc": float(auc),
            "url": URL.format(name),
            "source": "MedMNIST v2 (Yang et al., Scientific Data 2023), Zenodo record 10519652"}

os.makedirs("data_cache/medmnist", exist_ok=True)
out = {}
if os.path.exists("results/medmnist3d.json"):
    out = json.load(open("results/medmnist3d.json"))
for name in SUBSETS:
    if name in out:
        continue
    t0 = time.time()
    path = f"data_cache/medmnist/{name}.npz"
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            urllib.request.urlretrieve(URL.format(name), path)
        r = run_subset(name)
        out[name] = r
        json.dump(out, open("results/medmnist3d.json", "w"), indent=1)
        print(f"DONE {name} {time.time()-t0:.0f}s AUC {r['auc']:.3f}", flush=True)
    except Exception as e:
        print(f"FAIL {name} {type(e).__name__}: {str(e)[:80]}", flush=True)
print(f"MM3D_DONE analyzed_total={len(out)}", flush=True)
