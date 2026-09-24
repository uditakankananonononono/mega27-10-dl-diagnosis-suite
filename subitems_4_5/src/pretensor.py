"""Pre-tensorize image-folder datasets to uint8 .npy memmaps (one disk pass)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build(root: str, out_prefix: str, size: int, grayscale: bool = False):
    root = Path(root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    samples = []
    for idx, c in enumerate(classes):
        for ext in ("*.png", "*.jpeg", "*.jpg"):
            for p in sorted((root / c).glob(ext)):
                samples.append((p, idx))
    n = len(samples)
    ch = 1 if grayscale else 3
    x = np.lib.format.open_memmap(f"{out_prefix}_x.npy", mode="w+",
                                  dtype=np.uint8, shape=(n, ch, size, size))
    y = np.zeros(n, dtype=np.int64)
    ids = []
    for i, (p, label) in enumerate(samples):
        img = Image.open(p).convert("L" if grayscale else "RGB").resize(
            (size, size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        x[i] = arr[None] if grayscale else arr.transpose(2, 0, 1)
        y[i] = label
        ids.append(str(p.relative_to(root)))
        if i % 5000 == 0:
            print(i, flush=True)
    x.flush()
    np.save(f"{out_prefix}_y.npy", y)
    np.save(f"{out_prefix}_ids.npy", np.array(ids))
    np.save(f"{out_prefix}_classes.npy", np.array(classes))
    print("BUILT", out_prefix, n, "images", classes, flush=True)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], int(sys.argv[3]),
          grayscale=len(sys.argv) > 4 and sys.argv[4] == "gray")
