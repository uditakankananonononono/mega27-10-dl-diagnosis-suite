"""Hermetic test suite for MEGA27-10b. No network, no external datasets:
every fixture is synthetic and generated inside the test."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.data import ImageFolderDataset, TensorDataset
from src.common.models import (GlobalCNNClassifier, RegionGCNClassifier,
                               grid_adjacency, gcn_norm, param_count)
from src.common.train import train_model, predict_proba, full_metrics
from src.common.label_noise import (confident_joint, flag_label_errors,
                                    noise_summary)
from src.run_malaria import stratified_split


class TestGraphMath:
    def test_adjacency_symmetric_8neighbour(self):
        A = grid_adjacency(4)
        assert torch.allclose(A, A.t())
        # corner node has 3 neighbours, centre node has 8
        assert A[0].sum().item() == 3.0
        assert A[5].sum().item() == 8.0

    def test_gcn_norm_is_normalised(self):
        An = gcn_norm(grid_adjacency(3))
        n = An.shape[0]
        assert An.shape == (n, n)
        # eigenvalues of the normalised Laplacian lie in [-1, 1] =>
        # entries bounded and diagonal positive
        assert (An.diagonal() > 0).all()
        assert An.abs().max() <= 1.0


class TestModels:
    @pytest.mark.parametrize("in_ch,size", [(3, 64), (1, 128)])
    def test_forward_shapes(self, in_ch, size):
        for cls in (GlobalCNNClassifier, RegionGCNClassifier):
            m = cls(in_ch)
            out = m(torch.rand(2, in_ch, size, size))
            assert out.shape == (2, 2)
            assert torch.isfinite(out).all()

    def test_param_counts_small_enough_for_cpu(self):
        assert param_count(GlobalCNNClassifier(3)) < 300_000
        assert param_count(RegionGCNClassifier(1)) < 300_000

    def test_region_pooling_grid(self):
        m = RegionGCNClassifier(3, grid=4)
        fmap = torch.rand(2, 128, 8, 8)
        nodes = m.regions(fmap)
        assert nodes.shape == (2, 16, 64)


class _SyntheticDataset(torch.utils.data.Dataset):
    """Two Gaussian blobs in a 32x32 image; label = which blob is brighter."""

    def __init__(self, n=256, seed=0):
        rng = np.random.default_rng(seed)
        self.x = torch.rand(n, 3, 32, 32) * 0.3
        self.y = torch.randint(0, 2, (n,))
        for i in range(n):
            r, c = (4, 4) if self.y[i] == 0 else (24, 24)
            self.x[i, :, r:r + 4, c:c + 4] += 0.7

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


class TestTraining:
    def test_model_learns_separable_problem(self):
        ds = _SyntheticDataset(256)
        tr = torch.utils.data.Subset(ds, range(200))
        va = torch.utils.data.Subset(ds, range(200, 256))
        model = GlobalCNNClassifier(3)
        model, hist, best = train_model(model, tr, va, epochs=6, batch=32,
                                        seed=0, patience=4)
        probs, y = predict_proba(model, va, batch=32)
        m = full_metrics(probs, y)
        assert m["accuracy"] > 0.9
        assert len(hist) >= 1

    def test_metrics_keys(self):
        probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        y = np.array([0, 1, 0, 1])
        m = full_metrics(probs, y)
        for k in ("accuracy", "precision", "recall_sensitivity", "specificity",
                  "f1", "roc_auc", "confusion_matrix", "n"):
            assert k in m
        assert m["accuracy"] == 1.0


class TestLabelNoise:
    def _planted(self, n=2000, noise=0.1, seed=0):
        rng = np.random.default_rng(seed)
        y_true = rng.integers(0, 2, n)
        flip = rng.random(n) < noise
        y_given = np.where(flip, 1 - y_true, y_true)
        probs = np.zeros((n, 2))
        probs[:, 1] = np.clip(0.15 + 0.7 * y_true + rng.normal(0, 0.05, n), 0, 1)
        probs[:, 0] = 1 - probs[:, 1]
        return probs, y_given, flip

    def test_recovers_planted_flips(self):
        probs, y_given, flip = self._planted()
        flags = flag_label_errors(probs, y_given)
        recall = (flags & flip).sum() / flip.sum()
        fp_rate = (flags & ~flip).sum() / (~flip).sum()
        assert recall > 0.8
        assert fp_rate < 0.05

    def test_confident_joint_counts(self):
        probs, y_given, _ = self._planted(n=500)
        C = confident_joint(probs, y_given)
        assert C.shape == (2, 2)
        s = noise_summary(probs, y_given)
        assert 0.0 <= s["estimated_noise_rate"] <= 1.0


class TestData:
    def test_image_folder_dataset(self, tmp_path):
        rng = np.random.default_rng(0)
        for cname in ("a", "b"):
            d = tmp_path / cname
            d.mkdir()
            for i in range(4):
                arr = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
                Image.fromarray(arr).save(d / f"{i}.png")
        ds = ImageFolderDataset(tmp_path, 16, train=False)
        assert len(ds) == 8
        x, y = ds[0]
        assert x.shape == (3, 16, 16)
        assert ds.sample_id(0).startswith("a/0.png#")
        # augment path
        dst = ImageFolderDataset(tmp_path, 16, train=True, seed=1)
        xt, _ = dst[0]
        assert xt.shape == (3, 16, 16)

    def test_stratified_split(self):
        labels = np.array([0] * 80 + [1] * 20)
        tr, va, te = stratified_split(labels, 0)
        assert len(set(tr) & set(va)) == 0 and len(set(tr) & set(te)) == 0
        assert len(tr) + len(va) + len(te) == 100
        assert (labels[te] == 1).mean() == pytest.approx(0.2, abs=0.1)


class TestTensorDataset:
    def test_roundtrip(self):
        x = torch.rand(5, 3, 8, 8); y = torch.tensor([0, 1, 0, 1, 0])
        ds = TensorDataset(x, y)
        assert len(ds) == 5
        xi, yi = ds[3]
        assert yi == 1
