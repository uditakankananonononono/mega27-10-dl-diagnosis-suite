"""MedMNIST arm: 12 published medical-image benchmark subsets, real CNNs."""
import json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
import torch; torch.set_num_threads(2)
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import medmnist
from medmnist import INFO

SUBSETS = ["pathmnist", "chestmnist", "dermamnist", "octmnist",
           "pneumoniamnist", "retinamnist", "breastmnist", "bloodmnist",
           "tissuemnist", "organamnist", "organcmnist", "organsmnist"]

class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, n_classes))
    def forward(self, x):
        return self.f(x)

def run_subset(name, epochs=3, seed=0):
    info = INFO[name]
    z = np.load(f"data_cache/medmnist/{name}.npz")
    tr_imgs, tr_labels = z["train_images"], z["train_labels"]
    te_imgs, te_labels = z["test_images"], z["test_labels"]
    if tr_imgs.ndim == 4:  # RGB subsets -> grayscale (chunked, no float64 blowup)
        def gray(a):
            out = np.empty(a.shape[:3], np.uint8)
            for i in range(0, len(a), 20000):
                out[i:i+20000] = a[i:i+20000].astype(np.float32).mean(axis=3)
            return out
        tr_imgs, te_imgs = gray(tr_imgs), gray(te_imgs)
    n_classes = len(info["label"])
    task = info["task"]
    torch.manual_seed(seed)
    model = SmallCNN(n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ytr = torch.tensor(tr_labels.squeeze(), dtype=torch.long)
    bs = 128
    ntr = len(tr_imgs)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(ntr)
        for i in range(0, ntr, bs):
            idx = perm[i:i + bs]
            xb = torch.tensor(tr_imgs[idx.numpy()], dtype=torch.float32).unsqueeze(1) / 255.0
            logits = model(xb)
            if task == "multi-label, binary-class":
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, ytr[idx].float() if ytr.dim() > 1 else
                    nn.functional.one_hot(ytr[idx], n_classes).float())
            else:
                loss = nn.functional.cross_entropy(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    yte = te_labels.squeeze()
    with torch.no_grad():
        logits = torch.cat([model(torch.tensor(te_imgs[i:i + 256], dtype=torch.float32).unsqueeze(1) / 255.0)
                            for i in range(0, len(te_imgs), 256)])
    prob = torch.softmax(logits, dim=1).numpy()
    if task == "multi-label, binary-class":
        auc = roc_auc_score(yte, prob, average="macro")
    elif n_classes == 2:
        auc = roc_auc_score(yte, prob[:, 1])
    else:
        yoh = np.eye(n_classes)[yte]
        auc = roc_auc_score(yoh, prob, average="macro", multi_class="ovr")
    return {"subset": name, "task": task, "n_classes": n_classes,
            "n_train": len(tr_imgs), "n_test": len(te_imgs), "auc": float(auc),
            "source": "MedMNIST v2 (Yang et al., Scientific Data 2023), Zenodo"}

results = {}
if os.path.exists("results/medmnist.json"):
    results = json.load(open("results/medmnist.json"))
for name in SUBSETS:
    if name in results:
        continue
    t0 = time.time()
    try:
        r = run_subset(name)
        results[name] = r
        print(f"DONE {name} {time.time()-t0:.0f}s AUC {r['auc']:.4f}", flush=True)
        json.dump(results, open("results/medmnist.json", "w"), indent=1)
    except Exception as e:
        print(f"FAIL {name} {type(e).__name__}: {e}", flush=True)
print("MEDMNIST_DONE", flush=True)
