"""Optuna driver for pneumonia CNN lr search: optuna asks/tells, each trial
runs in a fresh subprocess (memory ceiling on 2GB box). Committed JSON."""
import json, time, sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pneumonia"

def main(n_trials=4):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=1))
    t0 = time.time()
    for _ in range(n_trials):
        trial = study.ask()
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        r = subprocess.run([sys.executable, str(ROOT / "src" / "optuna_pneu_worker.py"), str(lr)],
                           capture_output=True, text=True, timeout=1200)
        val = None
        for line in r.stdout.splitlines():
            if line.startswith("AUC_JSON "):
                val = json.loads(line[9:])["auc"]
        if val is None:
            print("WORKER_FAIL", r.returncode, r.stderr[-400:], flush=True)
            study.tell(trial, 0.5)
        else:
            print(f"trial lr={lr:.5f} auc={val:.4f}", flush=True)
            study.tell(trial, val)
    cur = json.load(open(OUT / "tuned_results.json"))["cnn"]
    out = {"n_trials": n_trials, "epochs_per_trial": 1,
           "subsample": "eighth of train, quarter val, batch 32; one fresh subprocess per trial (memory ceiling)",
           "best_val_auc": study.best_value, "best_params": study.best_params,
           "committed_tuned_cnn_test_auc": cur["roc_auc"],
           "trials": [{"value": t.value, "params": t.params} for t in study.trials],
           "secs": round(time.time() - t0, 1)}
    json.dump(out, open(OUT / "optuna_cnn.json", "w"), indent=1)
    print("OPTUNA_PNEU_DONE", flush=True)

if __name__ == "__main__":
    main()
