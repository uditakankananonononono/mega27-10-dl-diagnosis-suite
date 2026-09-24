"""Item 10.4: malaria parasite diagnosis. Benchmark + label-noise census."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.data import ImageFolderDataset, NpyDataset
from src.common.models import GlobalCNNClassifier, RegionGCNClassifier, param_count
from src.common.train import train_model, predict_proba, full_metrics, save_json
from src.common.label_noise import noise_summary, flag_label_errors

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "malaria" / "cell_images"
OUT = ROOT / "results" / "malaria"


def stratified_split(labels, seed, fracs=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    parts = [[], [], []]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx); n0 = int(fracs[0] * n); n1 = int(fracs[1] * n)
        parts[0].extend(idx[:n0]); parts[1].extend(idx[n0:n0 + n1]); parts[2].extend(idx[n0 + n1:])
    return [np.array(sorted(p)) for p in parts]


def oof_probabilities(train_idx, labels_all, *, kind, folds, epochs, seed, batch=64):
    """Out-of-fold predicted probs over train_idx using `kind` model."""
    probs = np.zeros((len(train_idx), 2), dtype=np.float64)
    order = np.argsort(train_idx)
    sorted_idx = np.asarray(train_idx)[order]
    sorted_labels = labels_all[sorted_idx]
    rng = np.random.default_rng(seed + 77)
    fold_of = np.zeros(len(sorted_idx), dtype=int)
    for c in np.unique(sorted_labels):
        pos = np.where(sorted_labels == c)[0]
        rng.shuffle(pos)
        fold_of[pos] = np.arange(len(pos)) % folds
    for f in range(folds):
        tr = sorted_idx[fold_of != f]; va = sorted_idx[fold_of == f]
        tr_ds = Subset(NpyDataset(str(ROOT / 'data' / 'malaria' / 'malaria48'), train=True, seed=seed + f), tr)
        va_ds = Subset(NpyDataset(str(ROOT / 'data' / 'malaria' / 'malaria48'), train=False), va)
        model = (GlobalCNNClassifier(3) if kind == "cnn" else RegionGCNClassifier(3))
        model, hist, best = train_model(model, tr_ds, va_ds, epochs=epochs,
                                        batch=batch, seed=seed + f, patience=3)
        p, _ = predict_proba(model, va_ds)
        probs[order[np.where(fold_of == f)[0]]] = p
        print(f"[oof fold {f}] val_loss={best['val_loss']:.4f}", flush=True)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["baseline", "census", "cleaned"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    eval_ds = NpyDataset(str(ROOT / "data" / "malaria" / "malaria48"), train=False)
    labels_all = np.asarray(eval_ds.y)
    tr_i, va_i, te_i = stratified_split(labels_all, args.seed)
    save_json({"train": tr_i.tolist(), "val": va_i.tolist(), "test": te_i.tolist(),
               "seed": args.seed, "classes": ["Parasitized", "Uninfected"]},
              OUT / "split.json")

    if args.phase == "baseline":
        results = {}
        for kind, cls in [("cnn", GlobalCNNClassifier), ("gcn", lambda in_ch: RegionGCNClassifier(in_ch, grid=3))]:
            tr_ds = Subset(NpyDataset(str(ROOT / 'data' / 'malaria' / 'malaria48'), train=True, seed=args.seed), tr_i)
            va_ds = Subset(eval_ds, va_i); te_ds = Subset(eval_ds, te_i)
            model = cls(3)
            t0 = time.time()
            model, hist, best = train_model(model, tr_ds, va_ds,
                                            epochs=args.epochs, seed=args.seed)
            probs, y = predict_proba(model, te_ds)
            m = full_metrics(probs, y)
            m["params"] = param_count(model); m["train_secs"] = round(time.time() - t0, 1)
            m["history"] = hist
            results[kind] = m
            save_json({"probs": probs.tolist(), "labels": y.tolist()},
                      OUT / f"test_probs_{kind}.json")
            print(f"[baseline {kind}] acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f}", flush=True)
        save_json(results, OUT / "baseline_results.json")
        print("BASELINE_DONE", flush=True)

    elif args.phase == "census":
        probs = oof_probabilities(tr_i, labels_all, kind="cnn", folds=3,
                                  epochs=args.epochs, seed=args.seed)
        y = labels_all[tr_i]
        save_json({"probs": probs.tolist()}, OUT / "oof_probs_train.json")
        summary = noise_summary(probs, y)
        flags = flag_label_errors(probs, y)
        flagged = [eval_ds.sample_id(int(tr_i[i])) for i in np.where(flags)[0]]
        save_json({"summary": summary, "flagged_ids": flagged},
                  OUT / "label_noise_census.json")
        print(f"[census] noise_rate={summary['estimated_noise_rate']:.4f} "
              f"flagged={len(flagged)}", flush=True)
        print("CENSUS_DONE", flush=True)

    elif args.phase == "cleaned":
        census = json.load(open(OUT / "label_noise_census.json"))
        flagged = set(census["flagged_ids"])
        keep = [i for i in tr_i if eval_ds.sample_id(int(i)) not in flagged]
        results = {}
        for kind, cls in [("cnn", GlobalCNNClassifier), ("gcn", lambda in_ch: RegionGCNClassifier(in_ch, grid=3))]:
            tr_ds = Subset(ImageFolderDataset(DATA, 48, train=True, seed=args.seed), keep)
            va_ds = Subset(eval_ds, va_i); te_ds = Subset(eval_ds, te_i)
            model = cls(3)
            model, hist, best = train_model(model, tr_ds, va_ds,
                                            epochs=args.epochs, seed=args.seed)
            probs, y = predict_proba(model, te_ds)
            m = full_metrics(probs, y)
            results[kind] = m
            print(f"[cleaned {kind}] acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f}", flush=True)
        results["removed"] = len(flagged)
        save_json(results, OUT / "cleaned_results.json")
        print("CLEANED_DONE", flush=True)


if __name__ == "__main__":
    main()
