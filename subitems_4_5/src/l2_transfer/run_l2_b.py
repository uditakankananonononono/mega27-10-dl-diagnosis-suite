#!/usr/bin/env python3
"""PREREG_PNEUMONIA_L2 Model B: frozen TorchXRayVision densenet121-res224-all + linear head.
Locked protocol: official Kermany splits, threshold chosen on VAL only then frozen,
single test evaluation per seed, 3 seeds. Features cached per split (encoder frozen)."""
import json, sys, time
import numpy as np, torch, torchxrayvision as xrv
from pathlib import Path
from PIL import Image
from torchvision import transforms

ROOT = Path.home() / "work/repos/mega27-10-dl-diagnosis-suite/subitems_4_5"
DATA = Path.home() / "work/data/kermany_cxr/chest_xray"
OUT = ROOT / "results/l2_b"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "feat_cache"; CACHE.mkdir(exist_ok=True)
SEEDS = [11, 23, 37]

tf = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((224, 224)), transforms.ToTensor(),
])

def load_split(split):
    imgs, labels = [], []
    for cls, y in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        for p in sorted((DATA / split / cls).glob("*.jpeg")):
            im = tf(Image.open(p).convert("L")).numpy()[0]
            imgs.append(xrv.datasets.normalize(im * 255.0, 255.0))
            labels.append(y)
    return np.array(imgs, dtype=np.float32), np.array(labels, dtype=np.int64)

def extract(model, split):
    f = CACHE / f"{split}_feat.npy"; l = CACHE / f"{split}_lab.npy"
    if f.exists():
        return np.load(f), np.load(l)
    x, y = load_split(split)
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            b = torch.from_numpy(x[i:i+32])
            out = model.features(b)          # (B,1024,7,7)
            feats.append(out.mean(dim=(2, 3)).numpy())
    F = np.concatenate(feats)
    np.save(f, F); np.save(l, y)
    return F, y

def fit_linear(F, y, seed, steps=1500, lr=0.05):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    X = torch.from_numpy(F).float(); Y = torch.from_numpy(y).float()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    w = torch.zeros(1024, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for i in range(steps):
        idx = torch.randint(0, len(Xn), (256,), generator=g)
        logit = Xn[idx] @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return w, b, mu, sd

def predict(F, w, b, mu, sd):
    Xn = (torch.from_numpy(F).float() - mu) / sd
    return torch.sigmoid(Xn @ w + b).numpy()

def main():
    t0 = time.time()
    model = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    Ftr, ytr = extract(model, "train")
    Fva, yva = extract(model, "val")
    Fte, yte = extract(model, "test")
    results = []
    for seed in SEEDS:
        w, b, mu, sd = fit_linear(Ftr, ytr, seed)
        pv = predict(Fva, w, b, mu, sd); pt = predict(Fte, w, b, mu, sd)
        # threshold on VAL only (Youden-style grid), frozen for test
        ths = np.linspace(0.05, 0.95, 181)
        th = max(ths, key=lambda t: ((pv >= t) == yva).mean())
        acc = float(((pt >= th) == yte).mean())
        from sklearn.metrics import roc_auc_score, brier_score_loss
        auc = float(roc_auc_score(yte, pt)); br = float(brier_score_loss(yte, pt))
        r = dict(seed=int(seed), val_thr=float(th), test_acc=acc, test_auc=auc, test_brier=br)
        results.append(r); print("SEED", r, flush=True)
        (OUT / "l2_b_results.json").write_text(json.dumps(dict(model="B", results=results, wall_s=time.time()-t0), indent=2))
    accs = [r["test_acc"] for r in results]
    print("DONE mean_acc=%.4f sd=%.4f" % (float(np.mean(accs)), float(np.std(accs, ddof=1))))

if __name__ == "__main__":
    main()
