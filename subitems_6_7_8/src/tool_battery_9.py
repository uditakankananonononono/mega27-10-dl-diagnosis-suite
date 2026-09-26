"""Battery 9: torchmetrics multiclass cross-verification and pycm stats
for the neuro dedup (leakage-controlled) CNN."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"


def verify_neuro_dedup():
    import torch
    from torchmetrics import AUROC, Accuracy, CalibrationError
    d = np.load(OUT / "neuro" / "brain_mri_cnn_dedup_probs.npz")
    y = torch.from_numpy(d["yte"] if "yte" in d.files else d["y_true"]).long()
    P = torch.from_numpy(d["probs"]).float()
    auc = AUROC(task="multiclass", num_classes=4)(P, y)
    acc = Accuracy(task="multiclass", num_classes=4)(P, y)
    ece = CalibrationError(task="multiclass", num_classes=4, n_bins=15)(P, y)
    ref = json.load(open(OUT / "neuro" / "brain_mri_cnn_dedup_train.json"))
    final = ref["history"][-1]
    json.dump({"tool": "torchmetrics (multiclass)",
               "auroc_ovr_torchmetrics": round(float(auc), 4),
               "acc_torchmetrics": round(float(acc), 4),
               "acc_train_json": final["acc"],
               "ece_15bin": round(float(ece), 4),
               "verdict": "metrics cross-verified" if abs(float(acc) - final["acc"]) < 0.002 else "MISMATCH - investigate"},
              open(OUT / "neuro" / "neuro_dedup_metrics_crossverify.json", "w"), indent=1)


def pycm_neuro_dedup():
    from pycm import ConfusionMatrix
    d = np.load(OUT / "neuro" / "brain_mri_cnn_dedup_probs.npz")
    y = d["yte"] if "yte" in d.files else d["y_true"]
    pred = d["probs"].argmax(1)
    cm = ConfusionMatrix(y.tolist(), pred.tolist())
    names = ["glioma", "meningioma", "notumor", "pituitary"]
    per_class = {names[c] if c < len(names) else str(c):
                 {"TPR": round(float(cm.class_stat["TPR"][c]), 4),
                  "PPV": round(float(cm.class_stat["PPV"][c]), 4),
                  "F1": round(float(cm.class_stat["F1"][c]), 4)} for c in sorted(cm.classes)}
    json.dump({"tool": "pycm ConfusionMatrix (multiclass)",
               "dataset": "neuro dedup held-out probs",
               "overall_acc": round(float(cm.overall_stat["Overall ACC"]), 4),
               "kappa": round(float(cm.overall_stat["Kappa"]), 4),
               "per_class": per_class,
               "note": "per-class recall/precision on the leakage-controlled model"},
              open(OUT / "neuro" / "neuro_dedup_pycm_stats.json", "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    {"verify": verify_neuro_dedup, "pycm": pycm_neuro_dedup}[which]()
    print(which, "done", flush=True)
