"""Cross-verify every committed accuracy/AUC with torchmetrics (an
independent metric implementation) against the sklearn-computed values
in the committed results JSONs. Agreement of two implementations guards
against a metric bug driving a headline number."""
import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

ROOT = Path(__file__).resolve().parent.parent

out = {}
for disease in ("malaria", "pneumonia"):
    for kind in ("cnn", "gcn"):
        p = ROOT / "results" / disease / f"test_probs_{kind}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        probs = torch.tensor(np.array(d["probs"], dtype=float)[:, 1])
        y = torch.tensor(np.array(d["labels"], dtype=int))
        acc = float(BinaryAccuracy()(probs, y))
        auc = float(BinaryAUROC()(probs, y))
        committed = json.load(open(ROOT / "results" / disease / "baseline_results.json"))[kind]
        out[f"{disease}/{kind}"] = {
            "torchmetrics_acc": round(acc, 6), "sklearn_acc": round(committed["accuracy"], 6),
            "torchmetrics_auc": round(auc, 6), "sklearn_auc": round(committed["roc_auc"], 6),
            "acc_match": abs(acc - committed["accuracy"]) < 1e-4,
            "auc_match": abs(auc - committed["roc_auc"]) < 1e-4}
json.dump(out, open(ROOT / "results" / "metrics_crossverify.json", "w"), indent=2)
for k, v in out.items():
    print(k, "acc", v["torchmetrics_acc"], "vs", v["sklearn_acc"], v["acc_match"],
          "| auc", v["torchmetrics_auc"], "vs", v["sklearn_auc"], v["auc_match"])
