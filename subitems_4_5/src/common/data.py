"""Dataset loaders for the NIH malaria cell dataset and the Kermany chest X-ray
pneumonia dataset. Real files only; transforms are deterministic given a seed."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class ImageFolderDataset(Dataset):
    """Two-class image folder dataset with stable, content-hashed IDs.

    Layout: root/<class_name>/*.png|jpeg|jpg
    Classes are sorted alphabetically; label = index in sorted class list.
    """

    def __init__(self, root: str | Path, image_size: int, grayscale: bool = False,
                 train: bool = True, seed: int = 0,
                 transform: Optional[Callable] = None):
        self.root = Path(root)
        self.image_size = image_size
        self.grayscale = grayscale
        self.train = train
        self.seed = seed
        self.transform = transform
        classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if len(classes) != 2:
            raise ValueError(f"expected 2 class dirs under {root}, found {classes}")
        self.classes = classes
        self.samples: list[tuple[Path, int]] = []
        for idx, cname in enumerate(classes):
            for ext in ("*.png", "*.jpeg", "*.jpg"):
                for p in sorted((self.root / cname).glob(ext)):
                    self.samples.append((p, idx))
        if not self.samples:
            raise ValueError(f"no images under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def sample_id(self, i: int) -> str:
        """Stable ID: relative path + md5 of the file bytes (first 12 hex)."""
        p, _ = self.samples[i]
        return f"{p.relative_to(self.root)}#{file_md5(p)[:12]}"

    def _augment(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < 0.5:
            img = img[:, ::-1]
        if rng.random() < 0.5:
            img = img[::-1, :]
        k = int(rng.integers(0, 4))
        img = np.rot90(img, k, axes=(0, 1)).copy()
        return img

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        mode = "L" if self.grayscale else "RGB"
        img = Image.open(path).convert(mode).resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.train:
            rng = np.random.default_rng(self.seed * 1_000_003 + i)
            arr = self._augment(arr, rng)
        if self.grayscale:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(arr)), label


class TensorDataset(Dataset):
    """Wrap pre-materialized tensors (used by tests and cached tensors)."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        assert x.shape[0] == y.shape[0]
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


class NpyDataset(Dataset):
    """RAM-resident tensor dataset built by pretensor.py. The uint8 array is
    loaded once per prefix and shared across all instances (singleton cache)."""

    _cache: dict = {}

    @classmethod
    def _shared(cls, prefix: str):
        if prefix not in cls._cache:
            cls._cache[prefix] = (np.load(f"{prefix}_x.npy"),
                                  np.load(f"{prefix}_y.npy"),
                                  np.load(f"{prefix}_ids.npy"))
        return cls._cache[prefix]

    def __init__(self, prefix: str, train: bool = False, seed: int = 0):
        self.x, self.y, self.ids = self._shared(prefix)
        self.y = np.load(f"{prefix}_y.npy")
        self.ids = np.load(f"{prefix}_ids.npy")
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.y)

    def sample_id(self, i: int) -> str:
        return str(self.ids[i])

    def __getitem__(self, i: int):
        arr = np.asarray(self.x[i], dtype=np.float32) / 255.0
        if self.train:
            rng = np.random.default_rng(self.seed * 1_000_003 + i)
            if rng.random() < 0.5:
                arr = arr[:, :, ::-1]
            if rng.random() < 0.5:
                arr = arr[:, ::-1, :]
            k = int(rng.integers(0, 4))
            arr = np.rot90(arr, k, axes=(1, 2)).copy()
        return torch.from_numpy(np.ascontiguousarray(arr)), int(self.y[i])
