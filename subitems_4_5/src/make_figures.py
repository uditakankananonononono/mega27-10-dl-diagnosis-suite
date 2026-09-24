"""Generate paper figures from committed result JSONs. Every figure traces to
a results/<disease>/*.json file; nothing is hand-drawn."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def fig_training_curves(disease: str):
    res = json.load(open(ROOT / "results" / disease / "baseline_results.json"))
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind in ("cnn", "gcn"):
        h = res[kind]["history"]
        ax.plot([e["epoch"] for e in h], [e["val_loss"] for e in h],
                marker="o", label=f"{kind.upper()} val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("val cross-entropy")
    ax.set_title(f"{disease}: validation loss during training")
    ax.legend(); fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{disease}_training_curves.png", dpi=150)
    plt.close(fig)


def fig_confusion(disease: str):
    res = json.load(open(ROOT / "results" / disease / "baseline_results.json"))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, kind in zip(axes, ("cnn", "gcn")):
        cm = np.array(res[kind]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_title(f"{kind.upper()} acc={res[kind]['accuracy']:.3f}")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.suptitle(f"{disease}: test confusion matrices")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{disease}_confusion.png", dpi=150)
    plt.close(fig)


def fig_noise_census(disease: str):
    c = json.load(open(ROOT / "results" / disease / "label_noise_census.json"))
    s = c["summary"]
    C = np.array(s["confident_joint"])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(C, cmap="Oranges")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(C[i, j]), ha="center", va="center")
    ax.set_xlabel("confident prediction"); ax.set_ylabel("given label")
    ax.set_title(f"{disease}: confident joint "
                 f"(noise rate {s['estimated_noise_rate']:.3f})")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{disease}_confident_joint.png", dpi=150)
    plt.close(fig)


def fig_cleaned_delta(disease: str):
    base = json.load(open(ROOT / "results" / disease / "baseline_results.json"))
    clean = json.load(open(ROOT / "results" / disease / "cleaned_results.json"))
    kinds = ["cnn", "gcn"]
    b = [base[k]["accuracy"] for k in kinds]
    c = [clean[k]["accuracy"] for k in kinds]
    x = np.arange(2); w = 0.35
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(x - w / 2, b, w, label="trained on original labels")
    ax.bar(x + w / 2, c, w, label=f"trained after census ({clean['removed']} removed)")
    ax.set_xticks(x, [k.upper() for k in kinds]); ax.set_ylabel("test accuracy")
    ax.set_ylim(min(b + c) - 0.02, max(b + c) + 0.02)
    ax.set_title(f"{disease}: effect of label-noise removal")
    ax.legend(); fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{disease}_cleaned_delta.png", dpi=150)
    plt.close(fig)


FIGS = {"baseline": [fig_training_curves, fig_confusion],
        "census": [fig_noise_census], "cleaned": [fig_cleaned_delta]}

if __name__ == "__main__":
    disease, phase = sys.argv[1], sys.argv[2]
    for fn in FIGS[phase]:
        fn(disease)
        print("made", fn.__name__)
