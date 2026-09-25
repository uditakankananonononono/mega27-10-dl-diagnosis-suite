"""Optuna search for the pneumonia CNN (item 10.5): lr and batch on a
quarter-subsample, 6 trials x 1 epoch (speed directive). Committed JSON."""
import json, time, sys, gc
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common.data import NpyDataset
from src.common.models import GlobalCNNClassifier
from src.common.train import train_model, predict_proba

OUT = ROOT / "results" / "pneumonia"

def main(n_trials=6, epochs=1):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ds = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_train"), train=True, seed=0)
    ev = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_train"), train=False)
    split = json.load(open(OUT / "split.json"))
    tr_i = np.array(split["train"])[::4]
    va_i = np.array(split["val"])[::2]  # half val for search speed/memory

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        batch = trial.suggest_categorical("batch", [32, 64])
        torch.manual_seed(0)
        model = GlobalCNNClassifier(1)
        model, hist, best = train_model(model, Subset(ds, tr_i), Subset(ev, va_i),
                                        epochs=epochs, batch=batch, lr=lr,
                                        seed=0, patience=2)
        probs, y = predict_proba(model, Subset(ev, va_i))
        val = float(roc_auc_score(y, probs[:, 1]))
        del model, hist, best, probs, y
        gc.collect()
        return val

    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(objective, n_trials=n_trials)
    cur = json.load(open(OUT / "tuned_results.json"))["cnn"]
    out = {"n_trials": n_trials, "epochs_per_trial": epochs,
           "subsample": "quarter of train, half val (speed+memory directive)",
           "best_val_auc": study.best_value, "best_params": study.best_params,
           "committed_tuned_cnn_test_auc": cur["roc_auc"],
           "trials": [{"value": t.value, "params": t.params} for t in study.trials],
           "secs": round(time.time() - t0, 1)}
    json.dump(out, open(OUT / "optuna_cnn.json", "w"), indent=1)
    print("OPTUNA_PNEU_DONE", flush=True)

if __name__ == "__main__":
    main()
