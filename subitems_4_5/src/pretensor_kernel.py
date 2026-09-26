"""Kernel-variant pretensorizer for the R-M2 resize diagnostic (judge round 6 pivot).
Builds malaria48-sized uint8 .npy datasets from raw cell_images with a chosen resize
kernel. Sliced + resumable: --start/--count process a slice; sidecar .done files mark
finished slices. Variants: pil_bilinear (existing malaria48), pil_lanczos, cv2_area,
cv2_linear. Same image order as src/pretensor.py build() so ids align exactly."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

KERNELS = {
    "pil_bilinear": lambda im, s: np.asarray(im.resize((s, s), Image.BILINEAR)),
    "pil_lanczos": lambda im, s: np.asarray(im.resize((s, s), Image.LANCZOS)),
    "cv2_area": lambda im, s: cv2.resize(np.asarray(im), (s, s), interpolation=cv2.INTER_AREA),
    "cv2_linear": lambda im, s: cv2.resize(np.asarray(im), (s, s), interpolation=cv2.INTER_LINEAR),
}

def build_slice(root, out_prefix, size, kernel, start, count):
    root = Path(root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    samples = []
    for idx, c in enumerate(classes):
        for ext in ("*.png", "*.jpeg", "*.jpg"):
            for p in sorted((root / c).glob(ext)):
                samples.append((p, idx))
    n = len(samples)
    xp = f"{out_prefix}_x.npy"
    if start == 0 and not Path(xp).exists():
        x = np.lib.format.open_memmap(xp, mode="w+", dtype=np.uint8, shape=(n, 3, size, size))
        y = np.zeros(n, dtype=np.int64)
        ids = []
    else:
        x = np.lib.format.open_memmap(xp, mode="r+")
        y = np.load(f"{out_prefix}_y.npy") if Path(f"{out_prefix}_y.npy").exists() else np.zeros(n, dtype=np.int64)
        ids = list(np.load(f"{out_prefix}_ids.npy")) if Path(f"{out_prefix}_ids.npy").exists() else []
    fn = KERNELS[kernel]
    end = min(start + count, n)
    for i in range(start, end):
        p, label = samples[i]
        img = Image.open(p).convert("RGB")
        arr = fn(img, size).astype(np.uint8)
        x[i] = arr.transpose(2, 0, 1)
        y[i] = label
        if len(ids) <= i:
            ids.append(str(p.relative_to(root)))
    x.flush()
    np.save(f"{out_prefix}_y.npy", y)
    np.save(f"{out_prefix}_ids.npy", np.array(ids))
    np.save(f"{out_prefix}_classes.npy", np.array(classes))
    Path(f"{out_prefix}_{start}_{end}.done").write_text("ok")
    print(f"SLICE DONE {kernel} {start}-{end} of {n}", flush=True)

if __name__ == "__main__":
    build_slice(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
