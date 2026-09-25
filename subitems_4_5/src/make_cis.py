"""Wilson score confidence intervals (statsmodels) for every committed
accuracy number. Reads only committed results JSONs; writes
results/confidence_intervals.json. No retraining, no network."""
import json
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parent.parent

def acc_ci(cm, alpha=0.05):
    cm = [[int(v) for v in row] for row in cm]
    correct = cm[0][0] + cm[1][1]
    n = sum(sum(r) for r in cm)
    lo, hi = proportion_confint(correct, n, alpha=alpha, method="wilson")
    return {"correct": correct, "n": n, "accuracy": correct / n,
            "wilson95": [round(lo, 4), round(hi, 4)]}

out = {}
for disease in ("malaria", "pneumonia"):
    for phase in ("baseline_results", "cleaned_results", "tuned_results"):
        p = ROOT / "results" / disease / f"{phase}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        for kind in ("cnn", "gcn"):
            if kind in d and "confusion_matrix" in d[kind]:
                out[f"{disease}/{phase}/{kind}"] = acc_ci(d[kind]["confusion_matrix"])

dest = ROOT / "results" / "confidence_intervals.json"
json.dump(out, open(dest, "w"), indent=2)
for k, v in out.items():
    print(f"{k}: {v['accuracy']:.4f} [{v['wilson95'][0]:.4f}, {v['wilson95'][1]:.4f}] (n={v['n']})")
