"""captum Integrated Gradients saliency on a retrained pathmnist CNN +
optuna hyperparameter search for the wdbc MLP.
Output analyses/captum_optuna.json + figures/captum_pathmnist.png"""
import json, os, sys, warnings
sys.path.insert(0, "src")
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
out = {}

# ---- captum on pathmnist CNN ----
try:
    import torch; torch.set_num_threads(2)
    import torch.nn as nn
    from captum.attr import IntegratedGradients
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load("data_cache/medmnist/pathmnist.npz")
    tr, te = z["train_images"], z["test_images"]
    ytr = z["train_labels"].squeeze(); yte = z["test_labels"].squeeze()
    def gray(a):
        outg = np.empty(a.shape[:3], np.uint8)
        for i in range(0, len(a), 20000):
            outg[i:i+20000] = a[i:i+20000].astype(np.float32).mean(axis=3)
        return outg
    if tr.ndim == 4:
        tr, te = gray(tr), gray(te)
    class SmallCNN(nn.Module):
        def __init__(s, n):
            super().__init__()
            s.f = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, n))
        def forward(s, x): return s.f(x)
    ncls = len(set(ytr.tolist()))
    torch.manual_seed(0)
    model = SmallCNN(ncls)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    ntr = len(tr); bs = 256
    model.train()
    perm = torch.randperm(ntr)
    for i in range(0, min(ntr, 60000), bs):  # ~1 epoch cap
        idx = perm[i:i + bs]
        xb = torch.tensor(tr[idx.numpy()], dtype=torch.float32).unsqueeze(1) / 255.0
        loss = nn.functional.cross_entropy(model(xb), ytr_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    ig = IntegratedGradients(model)
    xb = torch.tensor(te[:8], dtype=torch.float32).unsqueeze(1) / 255.0
    with torch.no_grad():
        preds = model(xb).argmax(1)
    attr = ig.attribute(xb, target=preds, n_steps=32)
    fig, axes = plt.subplots(2, 8, figsize=(13, 3.6))
    for j in range(8):
        axes[0, j].imshow(te[j], cmap="gray"); axes[0, j].axis("off")
        axes[0, j].set_title(f"pred {int(preds[j])}", fontsize=7)
        axes[1, j].imshow(np.abs(attr[j, 0].numpy()), cmap="hot"); axes[1, j].axis("off")
    fig.suptitle("pathmnist: test images (top) and Integrated-Gradients saliency (bottom)")
    fig.tight_layout(); fig.savefig("figures/captum_pathmnist.png", dpi=130)
    out["captum_pathmnist"] = {"tool": "captum",
        "method": "IntegratedGradients (32 steps) on CNN retrained ~1 epoch, 8 test images",
        "figure": "figures/captum_pathmnist.png",
        "mean_abs_attr": float(np.abs(attr.numpy()).mean())}
    print("DONE captum", flush=True)
except Exception as e:
    out["captum_pathmnist"] = {"tool": "captum", "error": str(e)[:160]}
    print(f"FAIL captum {type(e).__name__}: {str(e)[:80]}", flush=True)

# ---- optuna MLP search on wdbc ----
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from diagbench.data.clinical import load_wdbc
    ds = load_wdbc()
    cv = StratifiedKFold(4, shuffle=True, random_state=0)
    def objective(trial):
        h1 = trial.suggest_int("h1", 16, 128)
        h2 = trial.suggest_int("h2", 0, 64)
        lr = trial.suggest_float("lr", 1e-4, 5e-2, log=True)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        hidden = (h1,) if h2 == 0 else (h1, h2)
        clf = make_pipeline(StandardScaler(),
            MLPClassifier(hidden, learning_rate_init=lr, alpha=alpha,
                          max_iter=300, random_state=0))
        return cross_val_score(clf, ds.X, ds.y, cv=cv, scoring="roc_auc",
                               n_jobs=1).mean()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=15)
    out["optuna_wdbc"] = {"tool": "optuna",
        "method": "15-trial TPE search over MLP hidden sizes/lr/alpha, 4-fold CV AUC on wdbc",
        "best_auc": float(study.best_value), "best_params": study.best_params,
        "committed_logreg_auc": 0.9929,
        "note": "comparison point: committed wdbc logreg baseline"}
    print("DONE optuna", flush=True)
except Exception as e:
    out["optuna_wdbc"] = {"tool": "optuna", "error": str(e)[:160]}
    print(f"FAIL optuna {type(e).__name__}: {str(e)[:80]}", flush=True)

json.dump(out, open("analyses/captum_optuna.json", "w"), indent=1)
print("CAPTUM_OPTUNA_DONE", flush=True)
