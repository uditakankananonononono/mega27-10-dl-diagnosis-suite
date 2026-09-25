"""Battery 4 (post-CNN): cleanlab label-issue census on held-out probs,
torchmetrics cross-verification of every reported metric, seaborn prob
figures, torchinfo architecture audit. Runs after train_cnn completes."""
import json, sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results"


def census_pcam():
    from cleanlab.filter import find_label_issues
    d = np.load(OUT / "cancer" / "pcam_cnn_probs.npz")
    y, p = d["y_true"], d["prob_pos"]
    probs2 = np.stack([1 - p, p], axis=1)
    issues = find_label_issues(y, probs2, return_indices_ranked_by="self_confidence")
    n_flag = int(len(issues))
    flags = [{"patch_id": f"pcam_test_{i:06d}", "given": int(y[i]),
              "p_pos": round(float(p[i]), 4)} for i in issues[:50]]
    json.dump({"tool": "cleanlab (Northcutt et al.)", "dataset": "pcam_test",
               "method": "find_label_issues on CNN held-out probs (model trained on official valid split)",
               "n_test": int(len(y)), "n_flagged": n_flag,
               "noise_rate_pct": round(100 * n_flag / len(y), 2),
               "top50_flags": flags,
               "note": "held-out flags are model-conditional candidates, not proven label errors"},
              open(OUT / "cancer" / "pcam_label_census.json", "w"), indent=1)


def torchmetrics_verify():
    import torch
    from torchmetrics import AUROC, Accuracy, CalibrationError
    d = np.load(OUT / "cancer" / "pcam_cnn_probs.npz")
    y = torch.from_numpy(d["y_true"]).long()
    p = torch.from_numpy(np.stack([1 - d["prob_pos"], d["prob_pos"]], axis=1)).float()
    auc = AUROC(task="binary")(p[:, 1], y)
    acc = Accuracy(task="binary", num_classes=2)(p, y)
    ece = CalibrationError(task="binary", n_bins=15)(p, y)
    ref = json.load(open(OUT / "cancer" / "pcam_cnn_train.json"))
    final = ref["history"][-1]
    json.dump({"tool": "torchmetrics",
               "auroc_torchmetrics": round(float(auc), 4),
               "auroc_sklearn": final["test_auc"],
               "auroc_abs_delta": round(abs(float(auc) - final["test_auc"]), 5),
               "acc_torchmetrics": round(float(acc), 4),
               "acc_train_json": final["test_acc"],
               "ece_15bin": round(float(ece), 4),
               "verdict": "metrics cross-verified" if abs(float(auc) - final["test_auc"]) < 0.002 else "MISMATCH - investigate"},
              open(OUT / "cancer" / "pcam_metrics_crossverify.json", "w"), indent=1)


def prob_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    d = np.load(OUT / "cancer" / "pcam_cnn_probs.npz")
    fig, ax = plt.subplots(figsize=(6, 4))
    for lbl, name in ((0, "non-tumor"), (1, "tumor")):
        sns.kdeplot(d["prob_pos"][d["y_true"] == lbl], ax=ax, label=name, fill=True, alpha=0.4)
    ax.set_xlabel("CNN P(tumor)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cancer" / "fig_pcam_cnn_prob_dist.png", dpi=110)
    json.dump({"tool": "seaborn", "figure": "fig_pcam_cnn_prob_dist.png"},
              open(OUT / "cancer" / "tool_battery_4_seaborn.json", "w"), indent=1)


def arch_audit():
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from torchinfo import summary
    import train_cnn
    model = train_cnn.small_cnn(3, 2, 96)
    model.load_state_dict(torch.load(OUT / "cancer" / "pcam_cnn.pt", weights_only=True))
    s = summary(model, input_size=(1, 3, 96, 96), verbose=0)
    json.dump({"tool": "torchinfo", "model": "small_cnn pcam",
               "total_params": s.total_params, "trainable_params": s.trainable_params,
               "summary_str": str(s)[:800]},
              open(OUT / "cancer" / "pcam_arch_audit.json", "w"), indent=1)


if __name__ == "__main__":
    which = sys.argv[1]
    {"census": census_pcam, "verify": torchmetrics_verify,
     "figure": prob_figure, "arch": arch_audit}[which]()
    print(which, "done")
