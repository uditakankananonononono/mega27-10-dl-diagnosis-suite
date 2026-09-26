"""R-M1 (PREREG_RM1.md): 5-fold patient-cluster CV, GlobalCNNClassifier from
scratch, per-epoch checkpoint resume. Usage: run_rm1.py <fold_idx>"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.data import NpyDataset
from src.common.models import GlobalCNNClassifier
from src.common.train import full_metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "malaria" / "rm1"
OUT.mkdir(exist_ok=True)

def folds_assignment():
    ids = np.load(ROOT / "data" / "malaria" / "malaria48_ids.npy")
    codes = sorted({s.split("/")[1].split("_IMG")[0] for s in ids})
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(codes))
    folds = [[] for _ in range(5)]
    for i, c in enumerate(perm):
        folds[i % 5].append(codes[c])
    return ids, [set(f) for f in folds]

def main(fold):
    ids, folds = folds_assignment()
    code_of = np.array([s.split("/")[1].split("_IMG")[0] for s in ids])
    te_idx = np.where(np.isin(code_of, list(folds[fold])))[0]
    tr_clusters = set().union(*[folds[j] for j in range(5) if j != fold])
    tr_all = np.where(np.isin(code_of, list(tr_clusters)))[0]
    y = np.load(ROOT / "data" / "malaria" / "malaria48_y.npy")
    rng = np.random.default_rng(fold + 7)
    tr_codes = sorted(tr_clusters)
    rng.shuffle(tr_codes)
    va_codes = set(tr_codes[: max(1, len(tr_codes) // 10)])
    va_idx = tr_all[np.isin(code_of[tr_all], list(va_codes))]
    tr_idx = tr_all[~np.isin(code_of[tr_all], list(va_codes))]
    ckpt = OUT / f"fold{fold}_ckpt.pt"
    res_p = OUT / f"fold{fold}_result.json"
    if res_p.exists():
        print("fold", fold, "already done", flush=True)
        return
    ds_tr_full = NpyDataset(str(ROOT / "data" / "malaria" / "malaria48"), train=True, seed=fold + 42)
    ds_ev = NpyDataset(str(ROOT / "data" / "malaria" / "malaria48"), train=False)
    model = GlobalCNNClassifier(3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    start_ep, start_b = 0, 0
    hist = []
    if ckpt.exists():
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_ep, start_b = st["epoch"], st.get("batch", 0)
        hist = st["hist"]
        if start_b == 0:
            start_ep += 1
        print("resumed fold", fold, "at epoch", start_ep, "batch", start_b, flush=True)
    torch.manual_seed(fold + 42)
    lossf = torch.nn.CrossEntropyLoss()
    for ep in range(start_ep, 8):
        t0 = time.time()
        model.train()
        # deterministic per-epoch batch order (resume-safe)
        gen = torch.Generator().manual_seed(1000 * fold + ep)
        loader = DataLoader(Subset(ds_tr_full, tr_idx), batch_size=64, shuffle=True,
                            num_workers=0, generator=gen)
        tot, nl = 0.0, 0
        b0 = start_b if ep == start_ep else 0
        for bi, (xb, yb) in enumerate(loader):
            if bi < b0:
                if (bi + 1) % 50 == 0:
                    print("ffwd", bi + 1, flush=True)
                continue
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(yb); nl += len(yb)
            if (bi + 1) % 50 == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "epoch": ep, "batch": bi + 1, "hist": hist}, ckpt)
        start_b = 0
        # val
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            vloader = DataLoader(Subset(ds_ev, va_idx), batch_size=256)
            for xb, yb in vloader:
                out = model(xb)
                vl += float(lossf(out, yb)) * len(yb); vn += len(yb)
        hist.append({"epoch": ep, "train_loss": tot / nl, "val_loss": vl / vn,
                     "wall_s": round(time.time() - t0, 1)})
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "batch": 0, "hist": hist}, ckpt)
        print("fold", fold, "ep", ep, hist[-1], flush=True)
    # test
    model.eval()
    probs = []
    with torch.no_grad():
        tloader = DataLoader(Subset(ds_ev, te_idx), batch_size=256)
        for xb, yb in tloader:
            probs.append(torch.softmax(model(xb), dim=1).numpy())
    p = np.concatenate(probs)
    m = full_metrics(y[te_idx], p)
    res = {"fold": fold, "n_test": int(len(te_idx)), "n_train": int(len(tr_idx)),
           "metrics": m, "hist": hist}
    json.dump(res, open(res_p, "w"), indent=1)
    print("FOLD DONE", fold, m, flush=True)

if __name__ == "__main__":
    main(int(sys.argv[1]))
