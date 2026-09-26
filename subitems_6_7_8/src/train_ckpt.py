"""Checkpointable micro-slice CNN trainer: survives this sandbox's process
kills. Trains in time-boxed slices (default 240s), checkpoints model+opt+
epoch+step to ckpt.pt, updates the results JSON after every slice. Each
wake relaunches until the epoch budget is done. Datasets are harvested to
.npy once (stream-and-resize, OOM-safe) and reused across restarts."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders
from train_cnn import small_cnn

OUT = Path(__file__).resolve().parent.parent / "results"
NP = Path("/home/sandbox/mega27-expansion/np")
NP.mkdir(exist_ok=True)
torch.set_num_threads(1)


def harvest_pcam(split, cap):
    p = NP / f"pcam_{split}_{cap}.npz"
    if p.exists():
        d = np.load(p); return d["X"], d["y"]
    X = np.empty((cap, 3, 96, 96), dtype=np.uint8)
    y = np.empty(cap, dtype=np.int64)
    for i, r in enumerate(loaders.iter_pcam(split)):
        if i >= cap:
            break
        X[i] = r["image"].transpose(2, 0, 1); y[i] = r["label"]
    np.savez(p, X=X, y=y)
    return X, y


def harvest_breakhis(size=64):
    from PIL import Image
    import csv
    p = NP / f"breakhis_{size}.npz"
    split_of = {}
    with open(OUT / "cancer" / "breakhis_patient_splits.csv") as f:
        for r in csv.DictReader(f):
            split_of[r["image_id"].replace("histology_slides/breast/", "")] = r["split"]
    if p.exists():
        d = np.load(p); return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    n_tr = sum(1 for v in split_of.values() if v == "train")
    n_te = len(split_of) - n_tr
    Xtr = np.empty((n_tr, 3, size, size), dtype=np.uint8)
    Xte = np.empty((n_te, 3, size, size), dtype=np.uint8)
    ytr = np.empty(n_tr, dtype=np.int64); yte = np.empty(n_te, dtype=np.int64)
    i_tr = i_te = 0
    for r in loaders.iter_breakhis():
        im = np.asarray(Image.fromarray(r["image"]).resize((size, size), Image.BILINEAR), dtype=np.uint8).transpose(2, 0, 1)
        if split_of.get(r["image_id"], "train") == "train":
            Xtr[i_tr] = im; ytr[i_tr] = r["label"]; i_tr += 1
        else:
            Xte[i_te] = im; yte[i_te] = r["label"]; i_te += 1
    np.savez(p, Xtr=Xtr[:i_tr], ytr=ytr[:i_tr], Xte=Xte[:i_te], yte=yte[:i_te])
    return Xtr[:i_tr], ytr[:i_tr], Xte[:i_te], yte[:i_te]


def harvest_neuro(size=64):
    from PIL import Image
    import pyarrow.parquet as pq
    p = NP / f"neuro_{size}.npz"
    if p.exists():
        d = np.load(p); return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    n_all = len(pq.read_table(loaders.DATA / "neuro" / "brain_tumor_mri_train.parquet", columns=["label"]).column("label"))
    idx = set(np.random.RandomState(42).permutation(n_all)[: int(0.8 * n_all)].tolist())
    k = int(0.8 * n_all)
    Xtr = np.empty((k, 3, size, size), dtype=np.uint8)
    Xte = np.empty((n_all - k, 3, size, size), dtype=np.uint8)
    ytr = np.empty(k, dtype=np.int64); yte = np.empty(n_all - k, dtype=np.int64)
    i_tr = i_te = 0
    for i, r in enumerate(loaders.iter_neuro()):
        im = np.asarray(Image.fromarray(r["image"]).resize((size, size), Image.BILINEAR), dtype=np.uint8).transpose(2, 0, 1)
        if i in idx:
            Xtr[i_tr] = im; ytr[i_tr] = r["label"]; i_tr += 1
        else:
            Xte[i_te] = im; yte[i_te] = r["label"]; i_te += 1
    np.savez(p, Xtr=Xtr[:i_tr], ytr=ytr[:i_tr], Xte=Xte[:i_te], yte=yte[:i_te])
    return Xtr[:i_tr], ytr[:i_tr], Xte[:i_te], yte[:i_te]


def batch(X, idx):
    return torch.from_numpy(X[idx].astype(np.float32) / 255.0)


def evaluate(model, Xte, yte, n_classes):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(Xte), 256):
            ps.append(torch.softmax(model(batch(Xte, np.arange(i, min(i + 256, len(Xte))))), 1).numpy())
    probs = np.concatenate(ps)
    pred = probs.argmax(1)
    out = {"acc": round(float(accuracy_score(yte, pred)), 4),
           "bacc": round(float(balanced_accuracy_score(yte, pred)), 4)}
    if n_classes == 2:
        out["auc"] = round(float(roc_auc_score(yte, probs[:, 1])), 4)
    return out, probs


def run(name, get_data, n_classes, size, epochs, budget_s, out_json, bench):
    t0 = time.time()
    Xtr, ytr, Xte, yte = get_data()
    ytr_t = torch.from_numpy(ytr)
    ckpt_p = Path(str(out_json).replace("_train.json", "_ckpt.pt"))
    model = small_cnn(3, n_classes, size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    state = {"epoch": 0, "step": 0, "perm": None, "hist": []}
    if ckpt_p.exists():
        c = torch.load(ckpt_p, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        state = c["state"]
        print(f"resumed at epoch {state['epoch']} step {state['step']}", flush=True)
    done = False
    while state["epoch"] < epochs and not done:
        if state["step"] == 0:
            state["perm"] = np.random.permutation(len(ytr))
        perm = state["perm"]
        model.train()
        tot, nb = 0.0, 0
        steps = list(range(0, len(ytr), 64))
        for si, i in enumerate(steps[state["step"]:], start=state["step"]):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = lossf(model(batch(Xtr, idx)), ytr_t[idx])
            loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
            state["step"] = si + 1
            if time.time() - t0 > budget_s:
                break
        if state["step"] >= len(steps):
            met, probs = evaluate(model, Xte, yte, n_classes)
            row = {"epoch": state["epoch"] + 1, "train_loss": round(tot / max(nb, 1), 4), **met,
                   "elapsed_s": round(time.time() - t0, 1)}
            state["hist"].append(row)
            print(name, row, flush=True)
            state["epoch"] += 1
            state["step"] = 0
            json.dump({"dataset": name, "n_train": int(len(ytr)), "n_test": int(len(yte)),
                       "benchmark_to_beat": bench, "history": state["hist"]},
                      open(out_json, "w"), indent=1)
            np.savez(str(out_json).replace("_train.json", "_probs.npz"), y_true=yte,
                     **({"prob_pos": probs[:, 1]} if n_classes == 2 else {"probs": probs}))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state}, ckpt_p)
        if time.time() - t0 > budget_s:
            done = True
    finished = state["epoch"] >= epochs
    print(name, "FINISHED" if finished else "SLICE-END", flush=True)
    return finished


if __name__ == "__main__":
    which = sys.argv[1]
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    if which == "pcam":
        run("pcam valid->test", lambda: (*harvest_pcam("valid", 6144), *harvest_pcam("test", 4096)),
            2, 96, 4, budget, OUT / "cancer" / "pcam_cnn_train.json",
            "published PCam CNN ~0.90+ AUC (Veeling et al. arXiv:1806.03962)")
    elif which == "breakhis":
        run("breakhis patient-level", lambda: harvest_breakhis(64),
            2, 64, 4, budget, OUT / "cancer" / "breakhis_cnn_train.json",
            "published BreakHis image-level ~83-90% by mag (Spanhol 2016; ours 224px mirror -> 64px, caveat)")
    elif which == "neuro_dedup":
        def _dedup():
            d = np.load(NP / "neuro_64_dedup.npz")
            return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
        run("brain MRI dedup 4-class", _dedup,
            4, 64, 4, budget, OUT / "neuro" / "brain_mri_cnn_dedup_train.json",
            "self-reference: leaky-run acc 0.9117 bacc 0.9108; dedup run is the honest leakage-controlled number")
    elif which == "neuro":
        run("brain tumor MRI 4-class", lambda: harvest_neuro(64),
            4, 64, 4, budget, OUT / "neuro" / "brain_mri_cnn_train.json",
            "Kaggle-mirror CNN reports ~95%+ record-level (source-verify before claiming win)")
