"""1D CNN over the feature axis of clinical/omics profiles.

Features are treated as a length-d sequence (one channel in); convolutions
learn local motif-like interactions between neighbouring measurements, which
for ordered omics features (e.g. genes in genomic order) is biologically
meaningful, and for clinical panels acts as a structured regularizer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=kernel,
                              padding=kernel // 2)
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class CNN1D(nn.Module):
    def __init__(self, d_in: int, channels: tuple = (16, 32), kernel: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        blocks, prev = [], 1
        for c in channels:
            blocks.append(ConvBlock(prev, c, kernel))
            prev = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(prev, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.d_in = d_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n, d) -> (n, 1, d)
        h = self.features(x.unsqueeze(1))
        h = h.mean(dim=-1)  # global average pooling over the feature axis
        return self.head(h).squeeze(-1)
