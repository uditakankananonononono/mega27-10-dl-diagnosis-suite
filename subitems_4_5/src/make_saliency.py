"""Captum Integrated Gradients saliency for the committed malaria CNN.
Picks a small, seeded sample of test cells per class, computes IG
attributions vs a black baseline, saves an overlay figure and a
quantified concentration stat (fraction of attribution mass in the
central cell region vs the border)."""
import json, sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.data import NpyDataset
from src.common.models import GlobalCNNClassifier

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "malaria"


def main():
    ds = NpyDataset(str(ROOT / "data" / "malaria" / "malaria48"), train=False)
    split = json.load(open(OUT / "split.json"))
    te = np.array(split["test"])
    model = GlobalCNNClassifier(3)
    model.load_state_dict(torch.load(OUT / "model_cnn.pt", map_location="cpu"))
    model.eval()
    ig = IntegratedGradients(model)

    rng = np.random.default_rng(0)
    picks = []
    for c in (0, 1):
        idx = te[np.asarray(ds.y)[te] == c]
        picks.extend(rng.choice(idx, 4, replace=False).tolist())

    stats = []
    fig, axes = plt.subplots(len(picks), 3, figsize=(7, 2.1 * len(picks)))
    for row, i in enumerate(picks):
        arr = np.asarray(ds.x[i], dtype=np.float32) / 255.0   # C,H,W
        x = torch.from_numpy(arr[None].copy())
        x.requires_grad_(True)
        target = int(ds.y[i])
        attr, _ = ig.attribute(x, target=target, return_convergence_delta=True,
                               n_steps=32)
        a = attr[0].abs().sum(0).detach().numpy()
        a /= a.max() + 1e-9
        h, w = a.shape
        cy, cx = h // 4, w // 4
        centre_mass = float(a[cy:-cy, cx:-cx].sum() / a.sum())
        stats.append({"idx": int(i), "class": target,
                      "centre_attribution_mass": round(centre_mass, 4)})
        img = arr.transpose(1, 2, 0)
        axes[row, 0].imshow(img); axes[row, 0].set_title(f"true={ds.classes[target]}")
        axes[row, 1].imshow(a, cmap="hot"); axes[row, 1].set_title("IG attribution")
        axes[row, 2].imshow(img); axes[row, 2].imshow(a, cmap="hot", alpha=0.45)
        axes[row, 2].set_title(f"overlay (centre {centre_mass:.2f})")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "malaria_saliency_ig.png", dpi=140)

    masses = [s["centre_attribution_mass"] for s in stats]
    summary = {"n_images": len(stats),
               "centre_mass_mean": round(float(np.mean(masses)), 4),
               "centre_mass_per_image": stats,
               "note": "centre = central 50% x 50% crop; uniform attention would give 0.25"}
    json.dump(summary, open(OUT / "saliency_ig.json", "w"), indent=2)
    print("centre_mass_mean", summary["centre_mass_mean"])


if __name__ == "__main__":
    main()
