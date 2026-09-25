"""Transfer learning for pneumonia (item 10.5 benchmark-break attempt):
timm ResNet18 ImageNet-pretrained, fine-tuned on cxr_train, evaluated on the
official 624-film test split. Same split/protocol as every committed result."""
import json, time, sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common.data import NpyDataset
from src.common.train import full_metrics, save_json

OUT = ROOT / "results" / "pneumonia"

class Wrap(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # timm resnet18, num_classes=2
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--subsample", type=int, default=1, help="use every Nth train image")
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(2)

    import timm
    model = Wrap(timm.create_model("resnet18", pretrained=True, num_classes=2))
    tr_full = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_train"), train=True, seed=0)
    ev = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_train"), train=False)
    test_ds = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_test"), train=False)
    split = json.load(open(OUT / "split.json"))
    tr_i = np.array(split["train"])[::args.subsample]
    va_i = np.array(split["val"])
    print(f"train n={len(tr_i)} val n={len(va_i)}", flush=True)

    tr_loader = DataLoader(Subset(tr_full, tr_i), batch_size=32, shuffle=True,
                           generator=torch.Generator().manual_seed(0))
    va_loader = DataLoader(Subset(ev, va_i), batch_size=64)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()
    best_auc, best_state = -1, None
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tl = 0.0; nb = 0
        for x, yb in tr_loader:
            opt.zero_grad()
            out = model(x)
            loss = lossf(out, yb)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for x, yb in va_loader:
                ps.append(torch.softmax(model(x), 1)[:, 1].numpy()); ys.append(yb.numpy())
        pv = np.concatenate(ps); yv = np.concatenate(ys)
        auc = roc_auc_score(yv, pv)
        print(f"  epoch {ep}: train_loss={tl/max(nb,1):.4f} val_auc={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)

    model.eval(); ps, ys = [], []
    te_loader = DataLoader(test_ds, batch_size=64)
    with torch.no_grad():
        for x, yb in te_loader:
            ps.append(torch.softmax(model(x), 1).numpy()); ys.append(yb.numpy())
    probs = np.concatenate(ps); y = np.concatenate(ys)
    m = full_metrics(probs, y)
    m["train_secs"] = round(time.time() - t0, 1)
    m["recipe"] = {"backbone": "timm resnet18 imagenet", "epochs": args.epochs,
                   "subsample": args.subsample, "lr": args.lr,
                   "train_n": len(tr_i), "best_val_auc": round(best_auc, 4)}
    save_json(m, OUT / "transfer_results.json")
    save_json({"probs": probs.tolist(), "labels": y.tolist()}, OUT / "test_probs_transfer.json")
    torch.save(model.state_dict(), OUT / "model_transfer.pt")
    print(f"[transfer] acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f}", flush=True)
    print("TRANSFER_DONE", flush=True)

if __name__ == "__main__":
    main()
