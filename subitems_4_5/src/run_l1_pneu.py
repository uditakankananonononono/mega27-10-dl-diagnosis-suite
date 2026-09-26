"""Rung L1 (preregistered in PREREG_IMPROVEMENT_LADDER.md): frozen ImageNet
resnet18 features + logistic head. OOF-frozen threshold, single test eval.
Resumable feature extraction (memmap + progress json)."""
import json, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def extract(prefix, out_path, batch=64):
    d = np.load(f"{prefix}_x.npy"); y = np.load(f"{prefix}_y.npy")
    n = len(d)
    prog_path = f"{out_path}.progress.json"
    done = 0
    if os.path.exists(prog_path):
        done = json.load(open(prog_path))["done"]
        F = np.lib.format.open_memmap(out_path, mode="r+")
    else:
        F = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n, 512))
    rn = models.resnet18(weights="IMAGENET1K_V1")
    feat = nn.Sequential(*list(rn.children())[:-1]).eval()
    with torch.no_grad():
        for i in range(done, n, batch):
            x = d[i:i+batch].astype(np.float32) / 255.0
            x = np.repeat(x, 3, axis=1)
            x = (x - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
            F[i:i+batch] = feat(torch.from_numpy(x)).numpy().reshape(-1, 512)
            F.flush()
            json.dump({"done": min(i+batch, n)}, open(prog_path, "w"))
            if i % (batch*20) == 0: print(i, flush=True)
    np.save(f"{out_path}_y.npy", y)
    os.remove(prog_path)
    print("EXTRACTED", out_path, (n, 512), flush=True)

if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
