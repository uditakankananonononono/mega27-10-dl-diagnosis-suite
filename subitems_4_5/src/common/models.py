"""CNN core + region-graph GNN head, implemented from scratch in torch.

RegionGCN: the CNN feature map (C x H x W) is pooled into a g x g grid of
region embeddings (nodes). Edges follow 8-neighbour grid adjacency. Two GCN
layers propagate region evidence; a readout MLP produces logits.

GCN math (paper derivation): A_hat = D^{-1/2}(A + I)D^{-1/2};
H^{(l+1)} = sigma(A_hat H^{(l)} W^{(l)}).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_adjacency(g: int) -> torch.Tensor:
    """8-neighbour adjacency for a g x g grid, row-major node order."""
    n = g * g
    A = torch.zeros(n, n)
    for r in range(g):
        for c in range(g):
            i = r * g + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < g and 0 <= cc < g:
                        A[i, rr * g + cc] = 1.0
    return A


def gcn_norm(A: torch.Tensor) -> torch.Tensor:
    n = A.shape[0]
    A_hat = A + torch.eye(n)
    deg = A_hat.sum(dim=1)
    d_inv_sqrt = torch.diag(deg.clamp(min=1e-8) ** -0.5)
    return d_inv_sqrt @ A_hat @ d_inv_sqrt


class CNNCore(nn.Module):
    """Compact conv trunk: in_ch -> 32 -> 64 -> 128 feature map."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                   # s/2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # s/4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                   # s/8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class GlobalCNNClassifier(nn.Module):
    """Baseline head: global average pool + linear (plain CNN reference)."""

    def __init__(self, in_ch: int = 3, n_classes: int = 2):
        super().__init__()
        self.core = CNNCore(in_ch)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        fmap = self.core(x)
        z = fmap.mean(dim=(2, 3))
        return self.fc(z)


class RegionGCNClassifier(nn.Module):
    """Hybrid: CNN core, region graph over the feature map, GCN reasoning."""

    def __init__(self, in_ch: int = 3, n_classes: int = 2, grid: int = 4,
                 hidden: int = 64):
        super().__init__()
        self.core = CNNCore(in_ch)
        self.grid = grid
        self.node_proj = nn.Linear(128, hidden)
        self.w1 = nn.Linear(hidden, hidden)
        self.w2 = nn.Linear(hidden, hidden)
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_classes))
        A = grid_adjacency(grid)
        self.register_buffer("A_norm", gcn_norm(A))

    def regions(self, fmap: torch.Tensor) -> torch.Tensor:
        """Pool C x H x W feature map into grid x grid region embeddings."""
        B, C, H, W = fmap.shape
        g = self.grid
        assert H % g == 0 and W % g == 0, f"feature map {H}x{W} not divisible by grid {g}"
        pooled = F.adaptive_avg_pool2d(fmap, (g, g))      # B x C x g x g
        nodes = pooled.flatten(2).transpose(1, 2)          # B x (g*g) x C
        return self.node_proj(nodes)                       # B x (g*g) x hidden

    def forward(self, x):
        fmap = self.core(x)
        H = self.regions(fmap)                             # B x N x hidden
        A = self.A_norm
        H = F.relu(torch.bmm(A.expand(H.shape[0], -1, -1), self.w1(H)))
        H = F.relu(torch.bmm(A.expand(H.shape[0], -1, -1), self.w2(H)))
        z = H.mean(dim=1)                                  # mean readout
        return self.readout(z)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
