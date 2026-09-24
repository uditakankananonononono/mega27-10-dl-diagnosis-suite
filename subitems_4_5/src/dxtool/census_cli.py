"""dx-census: run a label-noise census on ANY two-class image folder.

Usage: python -m src.dxtool.census_cli FOLDER --image-size 64 --epochs 4 --folds 3
Writes <folder>/label_noise_census.json + .csv with content-hashed IDs.
"""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common.data import ImageFolderDataset
from src.common.models import GlobalCNNClassifier
from src.common.train import train_model, predict_proba
from src.common.label_noise import noise_summary, flag_label_errors
from src.common.train import save_json


def main():
    ap = argparse.ArgumentParser(prog="dx-census")
    ap.add_argument("folder")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    folder = Path(args.folder)
    ds_eval = ImageFolderDataset(folder, args.image_size, train=False)
    labels = np.array([y for _, y in ds_eval.samples])
    n = len(labels)
    order = np.argsort(labels, kind="stable")
    fold_of = np.zeros(n, dtype=int)
    rng = np.random.default_rng(args.seed)
    pos = np.arange(n)
    rng.shuffle(pos)
    fold_of[pos] = np.arange(n) % args.folds

    probs = np.zeros((n, 2))
    from torch.utils.data import Subset
    for f in range(args.folds):
        tr = np.where(fold_of != f)[0]; va = np.where(fold_of == f)[0]
        tr_ds = Subset(ImageFolderDataset(folder, args.image_size, train=True,
                                          seed=args.seed + f), tr)
        va_ds = Subset(ds_eval, va)
        model = GlobalCNNClassifier(3)
        model, _, _ = train_model(model, tr_ds, va_ds, epochs=args.epochs,
                                  seed=args.seed + f, patience=2)
        p, _ = predict_proba(model, va_ds)
        probs[va] = p
        print(f"fold {f} done", flush=True)

    summary = noise_summary(probs, labels)
    flags = flag_label_errors(probs, labels)
    flagged = [ds_eval.sample_id(i) for i in np.where(flags)[0]]
    out = {"summary": summary, "flagged_ids": flagged}
    save_json(out, folder / "label_noise_census.json")
    with open(folder / "label_noise_census.csv", "w") as fh:
        fh.write("sample_id,given_label,p_class0,p_class1\n")
        for i in np.where(flags)[0]:
            fh.write(f"{ds_eval.sample_id(i)},{labels[i]},{probs[i,0]:.4f},{probs[i,1]:.4f}\n")
    print(f"noise rate estimate: {summary['estimated_noise_rate']:.4f} "
          f"({summary['estimated_label_errors']}/{summary['n']}); "
          f"{len(flagged)} IDs written")


if __name__ == "__main__":
    main()
