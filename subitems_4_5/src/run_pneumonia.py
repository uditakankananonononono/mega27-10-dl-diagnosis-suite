"""Item 10.5: pneumonia diagnosis from chest X-ray (Kermany 2018 official split).
Benchmark + label-noise census. Test set is the untouched official 624 images."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.data import ImageFolderDataset
from src.common.models import GlobalCNNClassifier, RegionGCNClassifier, param_count
from src.common.train import train_model, predict_proba, full_metrics, save_json
from src.common.label_noise import noise_summary, flag_label_errors

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "pneumonia" / "chest_xray" / "train"
TEST_DIR = ROOT / "data" / "pneumonia" / "chest_xray" / "test"
OUT = ROOT / "results" / "pneumonia"
IMG = 128


def stratified_val(labels, seed, frac=0.1):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    labels = np.asarray(labels)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        k = int(frac * len(idx))
        va.extend(idx[:k]); tr.extend(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(va))


def oof_probabilities(train_idx, labels_all, *, folds, epochs, seed, batch=32):
    probs = np.zeros((len(train_idx), 2))
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
        tr_ds = Subset(ImageFolderDataset(TRAIN_DIR, IMG, grayscale=True,
                                          train=True, seed=seed + f), tr)
        va_ds = Subset(ImageFolderDataset(TRAIN_DIR, IMG, grayscale=True,
                                          train=False), va)
        model = GlobalCNNClassifier(1)
        model, hist, best = train_model(model, tr_ds, va_ds, epochs=epochs,
                                        batch=batch, seed=seed + f, patience=2)
        p, _ = predict_proba(model, va_ds)
        probs[order[np.where(fold_of == f)[0]]] = p
        print(f"[oof fold {f}] val_loss={best['val_loss']:.4f}", flush=True)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["baseline", "census", "cleaned"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_eval = ImageFolderDataset(TRAIN_DIR, IMG, grayscale=True, train=False)
    test_ds = ImageFolderDataset(TEST_DIR, IMG, grayscale=True, train=False)
    labels_all = np.array([y for _, y in train_eval.samples])
    tr_i, va_i = stratified_val(labels_all, args.seed)
    save_json({"train": tr_i.tolist(), "val": va_i.tolist(), "seed": args.seed,
               "classes": train_eval.classes,
               "note": "val carved from official train; official 624-image test untouched"},
              OUT / "split.json")

    if args.phase == "baseline":
        results = {}
        for kind, cls in [("cnn", GlobalCNNClassifier), ("gcn", RegionGCNClassifier)]:
            tr_ds = Subset(ImageFolderDataset(TRAIN_DIR, IMG, grayscale=True,
                                              train=True, seed=args.seed), tr_i)
            va_ds = Subset(train_eval, va_i)
            model = cls(1)
            t0 = time.time()
            model, hist, best = train_model(model, tr_ds, va_ds,
                                            epochs=args.epochs, batch=32,
                                            seed=args.seed, patience=2)
            probs, y = predict_proba(model, test_ds)
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
        probs = oof_probabilities(tr_i, labels_all, folds=3,
                                  epochs=args.epochs, seed=args.seed)
        y = labels_all[tr_i]
        save_json({"probs": probs.tolist()}, OUT / "oof_probs_train.json")
        summary = noise_summary(probs, y)
        flags = flag_label_errors(probs, y)
        flagged = [train_eval.sample_id(int(tr_i[i])) for i in np.where(flags)[0]]
        save_json({"summary": summary, "flagged_ids": flagged},
                  OUT / "label_noise_census.json")
        print(f"[census] noise_rate={summary['estimated_noise_rate']:.4f} "
              f"flagged={len(flagged)}", flush=True)
        print("CENSUS_DONE", flush=True)

    elif args.phase == "cleaned":
        census = json.load(open(OUT / "label_noise_census.json"))
        flagged = set(census["flagged_ids"])
        keep = [i for i in tr_i if train_eval.sample_id(int(i)) not in flagged]
        results = {}
        for kind, cls in [("cnn", GlobalCNNClassifier), ("gcn", RegionGCNClassifier)]:
            tr_ds = Subset(ImageFolderDataset(TRAIN_DIR, IMG, grayscale=True,
                                              train=True, seed=args.seed), keep)
            va_ds = Subset(train_eval, va_i)
            model = cls(1)
            model, hist, best = train_model(model, tr_ds, va_ds,
                                            epochs=args.epochs, batch=32,
                                            seed=args.seed, patience=2)
            probs, y = predict_proba(model, test_ds)
            m = full_metrics(probs, y)
            results[kind] = m
            print(f"[cleaned {kind}] acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f}", flush=True)
        results["removed"] = len(flagged)
        save_json(results, OUT / "cleaned_results.json")
        print("CLEANED_DONE", flush=True)


if __name__ == "__main__":
    main()
