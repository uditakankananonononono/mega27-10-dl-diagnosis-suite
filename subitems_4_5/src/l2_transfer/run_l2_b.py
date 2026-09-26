#!/usr/bin/env python3
"""PREREG_PNEUMONIA_L2 Model B: frozen TorchXRayVision densenet121-res224-all + linear head.
Locked protocol: official Kermany splits, threshold chosen on VAL only then frozen,
single test evaluation per seed, 3 seeds. Chunked resumable feature extraction
(each ~512-image chunk cached; survives interruption; run across many short calls)."""
import argparse, json, time
import numpy as np, torch, torchxrayvision as xrv
torch.set_num_threads(2)
from pathlib import Path
from PIL import Image
from torchvision import transforms

ROOT = Path.home() / "work/repos/mega27-10-dl-diagnosis-suite/subitems_4_5"
DATA = Path.home() / "work/data/kermany_cxr/chest_xray"
OUT = ROOT / "results/l2_b"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "feat_cache"; CACHE.mkdir(exist_ok=True)
SEEDS = [11, 23, 37]
BATCH, CHUNK_BATCHES = 32, 4   # 128 images per chunk

tf = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((224, 224)), transforms.ToTensor(),
])

def list_split(split):
    paths, labels = [], []
    for cls, y in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        for p in sorted((DATA / split / cls).glob("*.jpeg")):
            paths.append(p); labels.append(y)
    return paths, np.array(labels, dtype=np.int64)

def get_model():
    return xrv.models.DenseNet(weights="densenet121-res224-all").eval()

def extract_chunks(model, split, max_chunks):
    paths, _ = list_split(split)
    n_chunks = (len(paths) + BATCH * CHUNK_BATCHES - 1) // (BATCH * CHUNK_BATCHES)
    done = 0
    for ci in range(n_chunks):
        f = CACHE / f"{split}_chunk_{ci:02d}.npy"
        if f.exists():
            continue
        lo = ci * BATCH * CHUNK_BATCHES
        hi = min(lo + BATCH * CHUNK_BATCHES, len(paths))
        feats = []
        with torch.no_grad():
            for i in range(lo, hi, BATCH):
                ims = np.stack([xrv.datasets.normalize(tf(Image.open(p).convert("L")).numpy()[0] * 255.0, 255.0) for p in paths[i:i+BATCH]])
                out = model.features(torch.from_numpy(ims.astype(np.float32))[:, None, :, :])
                feats.append(out.mean(dim=(2, 3)).numpy())
        np.save(f, np.concatenate(feats))
        done += 1
        print(f"chunk {split}/{ci} done ({hi-lo} imgs)", flush=True)
        if done >= max_chunks:
            return False
    return all((CACHE / f"{split}_chunk_{ci:02d}.npy").exists() for ci in range(n_chunks))

def load_split_features(split):
    paths, labels = list_split(split)
    n_chunks = (len(paths) + BATCH * CHUNK_BATCHES - 1) // (BATCH * CHUNK_BATCHES)
    F = np.concatenate([np.load(CACHE / f"{split}_chunk_{ci:02d}.npy") for ci in range(n_chunks)])
    return F, labels

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
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "run", "status"])
    ap.add_argument("--chunks", type=int, default=2)
    a = ap.parse_args()
    if a.mode == "status":
        for split in ["train", "val", "test"]:
            paths, _ = list_split(split)
            n = (len(paths) + BATCH * CHUNK_BATCHES - 1) // (BATCH * CHUNK_BATCHES)
            got = sum(1 for ci in range(n) if (CACHE / f"{split}_chunk_{ci:02d}.npy").exists())
            print(f"{split}: {got}/{n} chunks")
        return
    if a.mode == "extract":
        model = get_model()
        for split in ["train", "val", "test"]:
            if not extract_chunks(model, split, a.chunks):
                return
        print("ALL CHUNKS DONE", flush=True)
        return
    # mode run: requires full cache
    t0 = time.time()
    Ftr, ytr = load_split_features("train")
    Fva, yva = load_split_features("val")
    Fte, yte = load_split_features("test")
    from sklearn.metrics import roc_auc_score, brier_score_loss
    results = []
    for seed in SEEDS:
        w, b, mu, sd = fit_linear(Ftr, ytr, seed)
        pv = predict(Fva, w, b, mu, sd); pt = predict(Fte, w, b, mu, sd)
        ths = np.linspace(0.05, 0.95, 181)
        th = max(ths, key=lambda t: ((pv >= t) == yva).mean())
        acc = float(((pt >= th) == yte).mean())
        auc = float(roc_auc_score(yte, pt)); br = float(brier_score_loss(yte, pt))
        r = dict(seed=int(seed), val_thr=float(th), test_acc=acc, test_auc=auc, test_brier=br)
        results.append(r); print("SEED", r, flush=True)
    accs = [r["test_acc"] for r in results]
    out = dict(model="B", protocol="PREREG_PNEUMONIA_L2", results=results,
               mean_acc=float(np.mean(accs)), sd_acc=float(np.std(accs, ddof=1)), wall_s=time.time() - t0)
    (OUT / "l2_b_results.json").write_text(json.dumps(out, indent=2))
    print("DONE mean_acc=%.4f sd=%.4f" % (out["mean_acc"], out["sd_acc"]))

if __name__ == "__main__":
    main()
