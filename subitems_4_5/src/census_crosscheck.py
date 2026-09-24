"""Independent cross-check of the from-scratch label-noise census against the
published cleanlab library (Northcutt et al.'s reference implementation).
Agreement of two independent estimators strengthens the census claim."""
import json, sys
from pathlib import Path

import numpy as np
from cleanlab.filter import find_label_issues
from cleanlab.count import estimate_joint, compute_confident_joint

ROOT = Path(__file__).resolve().parent.parent


def main(disease: str):
    oof = json.load(open(ROOT / f"results/{disease}/oof_probs_train.json"))
    split = json.load(open(ROOT / f"results/{disease}/split.json"))
    census = json.load(open(ROOT / f"results/{disease}/label_noise_census.json"))
    probs = np.array(oof["probs"])
    if disease == "malaria":
        from src.common.data import ImageFolderDataset
        ds = ImageFolderDataset(ROOT / "data/malaria/cell_images", 64, train=False)
        labels_all = np.array([y for _, y in ds.samples])
    else:
        from src.common.data import ImageFolderDataset
        ds = ImageFolderDataset(ROOT / "data/pneumonia/chest_xray/train", 128,
                                grayscale=True, train=False)
        labels_all = np.array([y for _, y in ds.samples])
    y = labels_all[np.array(split["train"])]

    cl_issues = find_label_issues(y, probs, return_indices_ranked_by="self_confidence")
    mine = set(census["flagged_ids"])
    theirs = {ds.sample_id(int(np.array(split["train"])[i])) for i in cl_issues}
    inter = mine & theirs
    out = {
        "ours_flagged": len(mine), "cleanlab_flagged": len(theirs),
        "intersection": len(inter),
        "jaccard": len(inter) / max(len(mine | theirs), 1),
        "cleanlab_joint": estimate_joint(y, probs).tolist(),
        "only_ours_sample": sorted(mine - theirs)[:25],
        "only_cleanlab_sample": sorted(theirs - mine)[:25],
        "agreed_ids": sorted(inter),
    }
    json.dump(out, open(ROOT / f"results/{disease}/census_crosscheck.json", "w"), indent=2)
    print(f"[{disease}] ours={len(mine)} cleanlab={len(theirs)} "
          f"agree={len(inter)} jaccard={out['jaccard']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
