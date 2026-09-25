"""Validation experiments for the malaria label-noise census (ISEF judge R3):
recovery  - inject known flips, measure estimator precision/recall vs injected
negctrl   - shuffled labels: admissibility gate must REFUSE the census
stability - GCN OOF flags vs committed CNN flags (cross-architecture Jaccard)
sensitivity - estimated noise rate across folds/seed/epochs configs
Each run is one subprocess launched by the driver (memory isolation)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.data import NpyDataset
from src.common.models import GlobalCNNClassifier, RegionGCNClassifier
from src.common.train import train_model, predict_proba
from src.common.label_noise import noise_summary, flag_label_errors, admissibility_gate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "malaria"
VOUT = OUT / "validation"

def oof(sub_idx, labels_all, *, kind, folds, epochs, seed, tag, batch=64):
    probs = np.zeros((len(sub_idx), 2), dtype=np.float64)
    order = np.argsort(sub_idx)
    sorted_idx = np.asarray(sub_idx)[order]
    sorted_labels = labels_all[sorted_idx]
    rng = np.random.default_rng(seed + 77)
    fold_of = np.zeros(len(sorted_idx), dtype=int)
    for c in np.unique(sorted_labels):
        pos = np.where(sorted_labels == c)[0]
        rng.shuffle(pos)
        fold_of[pos] = np.arange(len(pos)) % folds
    ckpt_dir = VOUT / f"oof_{tag}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for f in range(folds):
        ck = ckpt_dir / f"fold{f}.npy"
        pos = order[np.where(fold_of == f)[0]]
        if ck.exists():
            probs[pos] = np.load(ck); print(f"[{tag} fold {f}] ckpt", flush=True); continue
        tr = sorted_idx[fold_of != f]; va = sorted_idx[fold_of == f]
        tr_ds = Subset(NpyDataset(str(ROOT/'data'/'malaria'/'malaria48'), train=True, seed=seed+f), tr)
        va_ds = Subset(NpyDataset(str(ROOT/'data'/'malaria'/'malaria48'), train=False), va)
        model = GlobalCNNClassifier(3) if kind == "cnn" else RegionGCNClassifier(3)
        model, hist, best = train_model(model, tr_ds, va_ds, epochs=epochs, batch=batch, seed=seed+f, patience=3)
        p, _ = predict_proba(model, va_ds)
        np.save(ck, p); probs[pos] = p
        print(f"[{tag} fold {f}] val_loss={best['val_loss']:.4f}", flush=True)
        del model, tr_ds, va_ds, p
    return probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=["recovery","negctrl","stability","sensitivity"])
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--kind", default="cnn")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subsample", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    eval_ds = NpyDataset(str(ROOT/'data'/'malaria'/'malaria48'), train=False)
    labels_all = np.asarray(eval_ds.y)
    tr_i = np.array(json.load(open(OUT/"split.json"))["train"])
    sub = tr_i[::a.subsample]
    census = json.load(open(OUT/"label_noise_census.json"))
    real_flags = set(census["flagged_ids"])
    exp = a.experiment
    if exp == "recovery":
        rng = np.random.default_rng(a.seed + 1000)
        k = int(round(a.rate * len(sub)))
        flip_pos = np.sort(rng.choice(len(sub), size=k, replace=False))
        noisy = labels_all.copy()
        noisy[sub[flip_pos]] = 1 - noisy[sub[flip_pos]]
        tag = f"rec{int(a.rate*100)}s{a.seed}"
        probs = oof(sub, noisy, kind=a.kind, folds=a.folds, epochs=a.epochs, seed=a.seed, tag=tag)
        y = noisy[sub]
        gate = admissibility_gate(probs, y)
        summ = noise_summary(probs, y)
        flags = flag_label_errors(probs, y)
        flag_idx = set(np.where(flags)[0].tolist())
        inj = set(flip_pos.tolist())
        tp = len(flag_idx & inj)
        res = {"experiment": exp, "injected_rate": a.rate, "n": len(sub), "injected": k,
               "seed": a.seed, "folds": a.folds, "epochs": a.epochs, "subsample": a.subsample,
               "gate": gate, "estimated_noise_rate": summ["estimated_noise_rate"],
               "n_flagged": int(flags.sum()),
               "precision_vs_injected": tp / max(1, len(flag_idx)),
               "recall_vs_injected": tp / max(1, len(inj)),
               "flag_ids": [eval_ds.sample_id(int(sub[i])) for i in np.where(flags)[0]],
               "overlap_with_real_249": len([1 for i in np.where(flags)[0] if eval_ds.sample_id(int(sub[i])) in real_flags]),
               "note": "flags are estimator output on labels with injected synthetic flips; precision/recall measured against injected set only",
               "secs": round(time.time()-t0,1)}
    elif exp == "negctrl":
        rng = np.random.default_rng(a.seed + 2000)
        shuffled = rng.permutation(labels_all)
        tag = f"negctrls{a.seed}"
        probs = oof(sub, shuffled, kind=a.kind, folds=a.folds, epochs=a.epochs, seed=a.seed, tag=tag)
        y = shuffled[sub]
        gate = admissibility_gate(probs, y)
        summ = noise_summary(probs, y)  # computed to show what the gate blocks
        res = {"experiment": exp, "n": len(sub), "seed": a.seed, "folds": a.folds,
               "epochs": a.epochs, "subsample": a.subsample, "gate": gate,
               "ungated_estimated_noise_rate": summ["estimated_noise_rate"],
               "expected": "gate.admissible == False on shuffled labels; ungated rate shows the failure mode the gate blocks",
               "secs": round(time.time()-t0,1)}
    elif exp == "stability":
        tag = f"stab_{a.kind}s{a.seed}"
        probs = oof(sub, labels_all, kind=a.kind, folds=a.folds, epochs=a.epochs, seed=a.seed, tag=tag)
        y = labels_all[sub]
        gate = admissibility_gate(probs, y)
        summ = noise_summary(probs, y)
        flags = flag_label_errors(probs, y)
        ids = [eval_ds.sample_id(int(sub[i])) for i in np.where(flags)[0]]
        idset = set(ids)
        res = {"experiment": exp, "kind": a.kind, "n": len(sub), "seed": a.seed,
               "folds": a.folds, "epochs": a.epochs, "subsample": a.subsample,
               "gate": gate, "estimated_noise_rate": summ["estimated_noise_rate"],
               "n_flagged": len(ids), "flag_ids": ids,
               "overlap_with_cnn_249": len(idset & real_flags),
               "jaccard_with_cnn_249": len(idset & real_flags) / max(1, len(idset | real_flags)),
               "secs": round(time.time()-t0,1)}
    else:  # sensitivity
        tag = f"sens_f{a.folds}e{a.epochs}s{a.seed}"
        probs = oof(sub, labels_all, kind=a.kind, folds=a.folds, epochs=a.epochs, seed=a.seed, tag=tag)
        y = labels_all[sub]
        gate = admissibility_gate(probs, y)
        summ = noise_summary(probs, y)
        flags = flag_label_errors(probs, y)
        ids = set(eval_ds.sample_id(int(sub[i])) for i in np.where(flags)[0])
        res = {"experiment": exp, "kind": a.kind, "n": len(sub), "seed": a.seed,
               "folds": a.folds, "epochs": a.epochs, "subsample": a.subsample,
               "gate": gate, "estimated_noise_rate": summ["estimated_noise_rate"],
               "n_flagged": len(ids), "overlap_with_cnn_249": len(ids & real_flags),
               "secs": round(time.time()-t0,1)}
    name = f"{exp}_{tag if exp!='stability' else f'{a.kind}s'+str(a.seed)}.json"
    json.dump(res, open(VOUT/name, "w"), indent=1)
    print("EXP_DONE " + name, flush=True)

if __name__ == "__main__":
    main()
