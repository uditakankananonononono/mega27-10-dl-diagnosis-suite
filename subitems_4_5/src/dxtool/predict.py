"""Single-image prediction with the committed trained weights."""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common.models import GlobalCNNClassifier, RegionGCNClassifier

SPECS = {
    "malaria":   dict(weights="results/malaria/model_gcn.pt", in_ch=3, size=48,
                      grid=3, classes=["parasitised", "uninfected"]),
    "pneumonia": dict(weights="results/pneumonia/model_gcn.pt", in_ch=1, size=128,
                      grid=4, classes=["normal", "pneumonia"]),
}


def load_image(path: str, size: int, grayscale: bool) -> torch.Tensor:
    mode = "L" if grayscale else "RGB"
    img = Image.open(path).convert(mode).resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr[None] if grayscale else arr.transpose(2, 0, 1)
    return torch.from_numpy(arr[None].copy())


def main(name: str):
    ap = argparse.ArgumentParser(prog=f"dx-{name}")
    ap.add_argument("command", choices=["predict"])
    ap.add_argument("image")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    spec = SPECS[name]
    root = Path(__file__).resolve().parent.parent.parent
    wpath = args.weights or str(root / spec["weights"])
    model = RegionGCNClassifier(spec["in_ch"], grid=spec["grid"])
    state = torch.load(wpath, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    x = load_image(args.image, spec["size"], spec["in_ch"] == 1)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0].tolist()
    print(f"{spec['classes'][1]} probability: {p[1]:.4f}")
    print(f"{spec['classes'][0]} probability: {p[0]:.4f}")


def malaria():
    main("malaria")


def pneumonia():
    main("pneumonia")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in SPECS:
        name = sys.argv.pop(1)
    else:
        name = "malaria"
    main(name)
