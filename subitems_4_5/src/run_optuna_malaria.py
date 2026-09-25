"""Optuna hyperparameter search for the malaria GCN head.
Question: do the hand-chosen GCN hyperparameters leave accuracy on the table?
Objective: validation AUC on a held-out slice of the train split.
8 trials x 2 epochs (speed directive), CPU-bounded. Committed: results/malaria/optuna_gcn.json
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.data import NpyDataset
from src.common.models import RegionGCNClassifier
from src.common.train import train_model, predict_proba
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "malaria"


def main(n_trials=8, epochs=2):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ds = NpyDataset(str(ROOT / "data" / "malaria" / "malaria48"), train=True, seed=0)
    ev = NpyDataset(str(ROOT / "data" / "malaria" / "malaria48"), train=False)
    split = json.load(open(OUT / "split.json"))
    tr_i = np.array(split["train"]); va_i = np.array(split["val"])
    labels = np.asarray(ev.y)

    def objective(trial):
        hidden = trial.suggest_categorical("hidden", [32, 64, 128])
        grid = trial.suggest_categorical("grid", [3, 4, 6])
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        torch.manual_seed(0)
        model = RegionGCNClassifier(3, grid=grid, hidden=hidden)
        model, hist, best = train_model(model, Subset(ds, tr_i), Subset(ev, va_i),
                                        epochs=epochs, batch=64, lr=lr,
                                        seed=0, patience=2)
        probs, y = predict_proba(model, Subset(ev, va_i))
        return float(roc_auc_score(y, probs))

    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    cur = json.load(open(OUT / "baseline_results.json"))["gcn"]
    out = {"n_trials": n_trials, "epochs_per_trial": epochs,
           "best_val_auc": study.best_value, "best_params": study.best_params,
           "current_gcn_test_auc": cur["roc_auc"],
           "trials": [{"value": t.value, "params": t.params} for t in study.trials],
           "train_secs": round(time.time() - t0, 1)}
    json.dump(out, open(OUT / "optuna_gcn.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "trials"}, indent=1))
    print("OPTUNA_DONE", flush=True)


if __name__ == "__main__":
    main()
