"""Graph neural networks over patient-similarity graphs (pure torch).

Population-graph diagnosis in the spirit of Parisot et al. (2018): each node
is a patient, edges join similar profiles, and labels propagate through the
graph. Implemented without torch_geometric so the math is explicit and the
dependency footprint stays small.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    """X' = activation( A_norm X W ) with A_norm = D^-1/2 (A+I) D^-1/2."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ (x @ self.weight) + self.bias


class GCN(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32, dropout: float = 0.4):
        super().__init__()
        self.gc1 = GCNLayer(d_in, hidden)
        self.gc2 = GCNLayer(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.gc1(x, a_norm))
        h = self.dropout(h)
        return self.gc2(h, a_norm).squeeze(-1)  # logits (n,)


class GATLayer(nn.Module):
    """Single-head graph attention (Velickovic et al. 2018), dense form."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d_in, d_out))
        self.a_src = nn.Parameter(torch.empty(d_out, 1))
        self.a_dst = nn.Parameter(torch.empty(d_out, 1))
        self.bias = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        h = x @ self.W                                   # (n, d_out)
        e_src = h @ self.a_src                           # (n, 1)
        e_dst = h @ self.a_dst                           # (n, 1)
        e = torch.nn.functional.leaky_relu(e_src + e_dst.T, 0.2)  # (n, n)
        neg = torch.finfo(e.dtype).min
        e = torch.where(adj_mask > 0, e, torch.full_like(e, neg))
        alpha = torch.softmax(e, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        return alpha @ h + self.bias


class GAT(nn.Module):
    def __init__(self, d_in: int, hidden: int = 16, dropout: float = 0.4):
        super().__init__()
        self.gat1 = GATLayer(d_in, hidden)
        self.gat2 = GATLayer(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.gat1(x, adj_mask))
        h = self.dropout(h)
        return self.gat2(h, adj_mask).squeeze(-1)
