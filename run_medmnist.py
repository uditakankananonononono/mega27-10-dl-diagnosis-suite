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
    if tr_imgs.ndim == 4:  # RGB subsets -> grayscale
        tr_imgs = tr_imgs.mean(axis=3).astype(np.uint8)
        te_imgs = te_imgs.mean(axis=3).astype(np.uint8)
    n_classes = len(info["label"])
    task = info["task"]
    torch.manual_seed(seed)
    model = SmallCNN(n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr = torch.tensor(tr_imgs, dtype=torch.float32).unsqueeze(1) / 255.0
    ytr = torch.tensor(tr_labels.squeeze(), dtype=torch.long)
    bs = 128
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            logits = model(Xtr[idx])
            if task == "multi-label, binary-class":
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, ytr[idx].float() if ytr.dim() > 1 else
                    nn.functional.one_hot(ytr[idx], n_classes).float())
            else:
                loss = nn.functional.cross_entropy(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    Xte = torch.tensor(te_imgs, dtype=torch.float32).unsqueeze(1) / 255.0
    yte = te_labels.squeeze()
    with torch.no_grad():
        logits = torch.cat([model(Xte[i:i + 256]) for i in range(0, len(Xte), 256)])
    prob = torch.softmax(logits, dim=1).numpy()
    if n_classes == 2:
        auc = roc_auc_score(yte, prob[:, 1])
    else:
        yoh = np.eye(n_classes)[yte]
        auc = roc_auc_score(yoh, prob, average="macro", multi_class="ovr")
    return {"subset": name, "task": task, "n_classes": n_classes,
            "n_train": len(tr_imgs), "n_test": len(te_imgs), "auc": float(auc),
            "source": "MedMNIST v2 (Yang et al., Scientific Data 2023), Zenodo"}

results = {}
for name in SUBSETS:
    t0 = time.time()
    try:
        r = run_subset(name)
        results[name] = r
        print(f"DONE {name} {time.time()-t0:.0f}s AUC {r['auc']:.4f}", flush=True)
        json.dump(results, open("results/medmnist.json", "w"), indent=1)
    except Exception as e:
        print(f"FAIL {name} {type(e).__name__}: {e}", flush=True)
print("MEDMNIST_DONE", flush=True)
