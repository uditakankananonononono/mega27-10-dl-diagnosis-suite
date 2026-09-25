"""Pneumonia judge-R1 ablations: (1) random-155 removal vs flagged-155 removal;
(2) census flag stability under different seeds. Official 624-image test untouched.
One seed per invocation (subprocess memory isolation); results accumulate in JSON."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.data import NpyDataset
from src.common.models import GlobalCNNClassifier, RegionGCNClassifier
from src.common.train import train_model, predict_proba, full_metrics, save_json
from src.common.label_noise import flag_label_errors, admissibility_gate
from src.run_pneumonia import stratified_val, TRAIN_DIR, OUT, ROOT

TEST_NPY = ROOT / "data" / "pneumonia" / "cxr_test"
TRAIN_NPY = ROOT / "data" / "pneumonia" / "cxr_train"


def oof_probs(ckpt_tag, train_idx, labels_all, *, folds, epochs, seed, batch=32):
    """Same as run_pneumonia.oof_probabilities but with a per-tag checkpoint dir."""
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
    ckpt_dir = OUT / f"oof_ckpt_{ckpt_tag}"
    ckpt_dir.mkdir(exist_ok=True)
    for f in range(folds):
        ckpt = ckpt_dir / f"fold{f}_probs.npy"
        if ckpt.exists():
            probs[order[np.where(fold_of == f)[0]]] = np.load(ckpt)
            print(f"[oof {ckpt_tag} fold {f}] loaded checkpoint", flush=True)
            continue
        tr = sorted_idx[fold_of != f]; va = sorted_idx[fold_of == f]
        tr_ds = Subset(NpyDataset(str(TRAIN_NPY), train=True, seed=seed + f), tr)
        va_ds = Subset(NpyDataset(str(TRAIN_NPY), train=False), va)
        model = GlobalCNNClassifier(1)
        model, hist, best = train_model(model, tr_ds, va_ds, epochs=epochs,
                                        batch=batch, seed=seed + f, patience=2)
        p, _ = predict_proba(model, va_ds)
        np.save(ckpt, p)
        probs[order[np.where(fold_of == f)[0]]] = p
        del model, tr_ds, va_ds, p
        print(f"[oof {ckpt_tag} fold {f}] val_loss={best['val_loss']:.4f}", flush=True)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["random-removal", "census-stability"])
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    train_eval = NpyDataset(str(TRAIN_NPY), train=False)
    test_ds = NpyDataset(str(TEST_NPY), train=False)
    labels_all = np.asarray(train_eval.y)
    tr_i, va_i = stratified_val(labels_all, 42)  # identical split as canonical runs

    if args.phase == "random-removal":
        census = json.load(open(OUT / "label_noise_census.json"))
        n_remove = len(census["flagged_ids"])
        rng = np.random.default_rng(args.seed)
        drop = set(rng.choice(tr_i, size=n_remove, replace=False).tolist())
        keep = [i for i in tr_i if i not in drop]
        out_path = OUT / "removal_ablation.json"
        res = json.load(open(out_path)) if out_path.exists() else {"runs": {}}
        run = {}
        for kind, cls in [("cnn", GlobalCNNClassifier),
                          ("gcn", lambda in_ch: RegionGCNClassifier(in_ch, grid=4))]:
            tr_ds = Subset(NpyDataset(str(TRAIN_NPY), train=True, seed=42), keep)
            va_ds = Subset(train_eval, va_i)
            model = cls(1)
            t0 = time.time()
            model, hist, best = train_model(model, tr_ds, va_ds, epochs=8, batch=32,
                                            seed=42, patience=2)
            probs, y = predict_proba(model, test_ds)
            m = full_metrics(probs, y)
            m["train_secs"] = round(time.time() - t0, 1)
            run[kind] = m
            del model, tr_ds, va_ds
            print(f"[random-removal seed={args.seed} {kind}] acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f}", flush=True)
        run["removed"] = n_remove
        res["runs"][str(args.seed)] = run
        res["protocol"] = (f"remove {n_remove} RANDOM train images (seed per run) vs flagged removal; "
                           "identical split (seed 42), training config (8 epochs, batch 32, train seed 42, "
                           "patience 2) and untouched official 624-image test as cleaned_results.json")
        save_json(res, out_path)
        print("RANDOM_REMOVAL_DONE", flush=True)

    else:  # census-stability
        probs = oof_probs(f"stab{args.seed}", tr_i, labels_all, folds=3, epochs=4, seed=args.seed)
        y = labels_all[tr_i]
        gate = admissibility_gate(probs, y)
        flags = flag_label_errors(probs, y)
        flagged = sorted(train_eval.sample_id(int(tr_i[i])) for i in np.where(flags)[0])
        canonical = set(json.load(open(OUT / "label_noise_census.json"))["flagged_ids"])
        inter = len(canonical & set(flagged)); union = len(canonical | set(flagged))
        out_path = OUT / "census_stability.json"
        res = json.load(open(out_path)) if out_path.exists() else {"runs": {}}
        res["runs"][str(args.seed)] = {
            "oof_accuracy": gate["oof_accuracy"], "admissible": gate["admissible"],
            "n_flags": len(flagged), "overlap_with_canonical_155": inter,
            "jaccard_vs_canonical": round(inter / union, 4) if union else None,
            "flagged_ids": flagged}
        res["protocol"] = ("re-run census OOF (3 folds, 4 epochs - reduced validation protocol) at a different seed; "
                           "compare flag set with canonical seed-42 census (155 flags)")
        save_json(res, out_path)
        print(f"[census-stability seed={args.seed}] flags={len(flagged)} overlap={inter} jaccard={inter/union:.4f}", flush=True)
        print("CENSUS_STABILITY_DONE", flush=True)


if __name__ == "__main__":
    main()
