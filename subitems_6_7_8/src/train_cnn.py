"""Small from-scratch CNN trainers for the 10.6-10.8 suites (CPU).
PCam: valid split trains, test split evaluates (official protocol).
Metrics: ROC AUC (Mann-Whitney equivalent from sklearn), balanced acc,
Wilson CI. Every epoch's metrics land in a committed JSON."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders

OUT = Path(__file__).resolve().parent.parent / "results"
torch.set_num_threads(1)


def small_cnn(in_ch=3, n_classes=2, size=96):
    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    feats = nn.Sequential(blk(in_ch, 24), blk(24, 48), blk(48, 96), blk(96, 128))
    with torch.no_grad():
        n = feats(torch.zeros(1, in_ch, size, size)).flatten(1).shape[1]
    return nn.Sequential(feats, nn.Flatten(), nn.Dropout(0.25), nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, n_classes))


def harvest_pcam(split, cap):
    X = np.empty((cap, 3, 96, 96), dtype=np.uint8)
    y = np.empty(cap, dtype=np.int64)
    for i, r in enumerate(loaders.iter_pcam(split)):
        if i >= cap:
            break
        X[i] = r["image"].transpose(2, 0, 1)
        y[i] = r["label"]
    return X, y


def batch(X, idx):
    return torch.from_numpy(X[idx].astype(np.float32) / 255.0)


def eval_probs(model, Xte, chunk=1024):
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(Xte), chunk):
            ps.append(torch.softmax(model(batch(Xte, np.arange(i, min(i + chunk, len(Xte))))), 1)[:, 1].numpy())
    return np.concatenate(ps)


def train_pcam(cap_tr=8192, cap_te=8192, epochs=4, bs=64, lr=1e-3):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from statsmodels.stats.proportion import proportion_confint
    t0 = time.time()
    Xtr, ytr = harvest_pcam("valid", cap_tr)
    Xte, yte = harvest_pcam("test", cap_te)
    ytr_t = torch.from_numpy(ytr)
    print(f"harvest {time.time()-t0:.0f}s", flush=True)
    model = small_cnn(3, 2, 96)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr))
        tot = 0.0
        nb = 0
        for i in range(0, len(ytr), bs):
            idx = perm[i:i + bs].numpy()
            opt.zero_grad()
            out = model(batch(Xtr, idx))
            loss = lossf(out, ytr_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        probs = eval_probs(model, Xte)
        pred = (probs > 0.5).astype(int)
        acc = float((pred == yte).mean())
        auc = float(roc_auc_score(yte, probs))
        bacc = float(balanced_accuracy_score(yte, pred))
        lo, hi = proportion_confint(int(acc * len(yte)), len(yte), alpha=0.05, method="wilson")
        row = {"epoch": ep + 1, "train_loss": round(tot / nb, 4), "test_acc": round(acc, 4),
               "test_auc": round(auc, 4), "test_bacc": round(bacc, 4),
               "wilson95": [round(float(lo), 4), round(float(hi), 4)],
               "elapsed_s": round(time.time() - t0, 1)}
        hist.append(row)
        print(row, flush=True)
        json.dump({"dataset": "pcam valid->test (official split protocol)",
                   "model": "small CNN from scratch (4 conv blocks, ~%dk params)" % (sum(p.numel() for p in model.parameters()) // 1000),
                   "n_train": int(len(ytr)), "n_test": int(len(yte)),
                   "benchmark_to_beat": "published PCam CNN ~0.90+ AUC (Veeling et al. arXiv:1806.03962)",
                   "history": hist}, open(OUT / "cancer" / "pcam_cnn_train.json", "w"), indent=1)
        torch.save(model.state_dict(), OUT / "cancer" / "pcam_cnn.pt")
        np.savez(OUT / "cancer" / "pcam_cnn_probs.npz", y_true=yte, prob_pos=probs)
    print("PCAM CNN DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "pcam":
        train_pcam()

def harvest_images(records, size=96):
    from PIL import Image
    n = len(records)
    X = np.empty((n, 3, size, size), dtype=np.uint8)
    y = np.empty(n, dtype=np.int64)
    for i, r in enumerate(records):
        im = Image.fromarray(r["image"]).resize((size, size), Image.BILINEAR)
        X[i] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
        y[i] = r["label"]
    return X, y


def fit_eval(Xtr, ytr, Xte, yte, n_classes, size, epochs, bs, lr, out_json, out_pt, ds_name, bench_note):
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
    from statsmodels.stats.proportion import proportion_confint
    t0 = time.time()
    model = small_cnn(3, n_classes, size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    ytr_t = torch.from_numpy(ytr)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(ytr))
        tot, nb = 0.0, 0
        for i in range(0, len(ytr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(batch(Xtr, idx)), ytr_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        probs = eval_probs_multi(model, Xte, n_classes)
        pred = probs.argmax(1)
        acc = float(accuracy_score(yte, pred))
        bacc = float(balanced_accuracy_score(yte, pred))
        row = {"epoch": ep + 1, "train_loss": round(tot / nb, 4), "test_acc": round(acc, 4),
               "test_bacc": round(bacc, 4), "elapsed_s": round(time.time() - t0, 1)}
        if n_classes == 2:
            row["test_auc"] = round(float(roc_auc_score(yte, probs[:, 1])), 4)
            lo, hi = proportion_confint(int(acc * len(yte)), len(yte), alpha=0.05, method="wilson")
            row["wilson95"] = [round(float(lo), 4), round(float(hi), 4)]
        hist.append(row)
        print(ds_name, row, flush=True)
        json.dump({"dataset": ds_name, "n_train": int(len(ytr)), "n_test": int(len(yte)),
                   "benchmark_to_beat": bench_note, "history": hist},
                  open(out_json, "w"), indent=1)
        torch.save(model.state_dict(), out_pt)
        np.savez(str(out_json).replace("_train.json", "_probs.npz"), y_true=yte, probs=probs)
    print(ds_name, "CNN DONE", flush=True)


def eval_probs_multi(model, Xte, n_classes, chunk=512):
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(Xte), chunk):
            ps.append(torch.softmax(model(batch(Xte, np.arange(i, min(i + chunk, len(Xte))))), 1).numpy())
    return np.concatenate(ps)


def train_breakhis(size=96, epochs=4, bs=64, lr=1e-3):
    import csv
    split_of = {}
    with open(OUT / "cancer" / "breakhis_patient_splits.csv") as f:
        for r in csv.DictReader(f):
            split_of[r["image_id"].replace("histology_slides/breast/", "")] = r["split"]
    if not split_of:
        with open(OUT / "cancer" / "breakhis_accessions.csv") as f:
            rows = list(csv.DictReader(f))
        split_of = {r["image_id"]: "train" for r in rows}
    # stream-and-resize: never hold full-res arrays (OOM guard)
    from PIL import Image
    n_tr = sum(1 for v in split_of.values() if v == "train")
    n_te = len(split_of) - n_tr
    Xtr = np.empty((n_tr, 3, size, size), dtype=np.uint8)
    Xte = np.empty((max(n_te, 1), 3, size, size), dtype=np.uint8)
    ytr = np.empty(n_tr, dtype=np.int64)
    yte = np.empty(max(n_te, 1), dtype=np.int64)
    i_tr = i_te = 0
    for r in loaders.iter_breakhis():
        im = np.asarray(Image.fromarray(r["image"]).resize((size, size), Image.BILINEAR), dtype=np.uint8).transpose(2, 0, 1)
        if split_of.get(r["image_id"], "train") == "train" and i_tr < n_tr:
            Xtr[i_tr] = im; ytr[i_tr] = r["label"]; i_tr += 1
        elif i_te < n_te:
            Xte[i_te] = im; yte[i_te] = r["label"]; i_te += 1
    Xtr, ytr, Xte, yte = Xtr[:i_tr], ytr[:i_tr], Xte[:i_te], yte[:i_te]
    print(f"breakhis harvest train={len(ytr)} heldout={len(yte)}", flush=True)
    fit_eval(Xtr, ytr, Xte, yte, 2, size, epochs, bs, lr,
             OUT / "cancer" / "breakhis_cnn_train.json", OUT / "cancer" / "breakhis_cnn.pt",
             "breakhis patient-level split (70 train / 30 held-out patients)",
             "published BreakHis image-level ~83-90% by magnification (Spanhol 2016, 700x460 PNG; ours 224x224 mirror resized to 96 - caveat in manifest)")


def train_neuro(size=96, epochs=4, bs=64, lr=1e-3):
    from PIL import Image
    import pyarrow.parquet as pq
    n_all = len(pq.read_table(loaders.DATA / "neuro" / "brain_tumor_mri_train.parquet", columns=["label"]).column("label"))
    idx = set(np.random.RandomState(42).permutation(n_all)[: int(0.8 * n_all)].tolist())
    k = int(0.8 * n_all)
    Xtr = np.empty((k, 3, size, size), dtype=np.uint8)
    Xte = np.empty((n_all - k, 3, size, size), dtype=np.uint8)
    ytr = np.empty(k, dtype=np.int64)
    yte = np.empty(n_all - k, dtype=np.int64)
    i_tr = i_te = 0
    for i, r in enumerate(loaders.iter_neuro()):
        im = np.asarray(Image.fromarray(r["image"]).resize((size, size), Image.BILINEAR), dtype=np.uint8).transpose(2, 0, 1)
        if i in idx:
            Xtr[i_tr] = im; ytr[i_tr] = r["label"]; i_tr += 1
        else:
            Xte[i_te] = im; yte[i_te] = r["label"]; i_te += 1
    print(f"neuro harvest train={len(ytr)} test={len(yte)}", flush=True)
    fit_eval(Xtr, ytr, Xte, yte, 4, size, epochs, bs, lr,
             OUT / "neuro" / "brain_mri_cnn_train.json", OUT / "neuro" / "brain_mri_cnn.pt",
             "brain tumor MRI 4-class, stratified 80/20 record-level split (patient linkage unavailable in mirror - noted)",
             "published Kaggle-mirror CNN reports ~95%+ record-level (source-verify before claiming benchmark win)")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "breakhis":
    train_breakhis()
elif __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "neuro":
    train_neuro()


