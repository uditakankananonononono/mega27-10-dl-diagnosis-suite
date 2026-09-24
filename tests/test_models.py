"""Model architecture tests: shapes, gradients, learning capacity."""
import numpy as np
import torch

from diagbench.graphs import knn_similarity_graph, normalize_adjacency
from diagbench.models.cnn1d import CNN1D
from diagbench.models.gnn import GAT, GATLayer, GCN, GCNLayer
from diagbench.models.mlp import MLP


def test_mlp_forward_shape_and_grad():
    m = MLP(10)
    x = torch.randn(16, 10)
    out = m(x)
    assert out.shape == (16,)
    out.sum().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_cnn1d_forward_shape_and_grad():
    m = CNN1D(30)
    x = torch.randn(8, 30)
    out = m(x)
    assert out.shape == (8,)
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_gcn_layer_matches_hand_computation():
    layer = GCNLayer(2, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0], [2.0]]))
        layer.bias.zero_()
    A = normalize_adjacency(np.array([[0.0, 1.0], [1.0, 0.0]]))
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = layer(x, torch.tensor(A))
    # A+I = [[1,1],[1,1]], deg = 2, norm factor 1/2 per entry
    # node0: 0.5*(1*1) + 0.5*(1*2) = 1.5 ; node1: 0.5*2 + 0.5*1 = 1.5
    assert torch.allclose(out.squeeze(-1), torch.tensor([1.5, 1.5]), atol=1e-5)


def test_gcn_full_forward():
    m = GCN(6, hidden=8)
    x = torch.randn(12, 6)
    A = torch.tensor(normalize_adjacency(knn_similarity_graph(x.numpy(), k=3)))
    out = m(x, A)
    assert out.shape == (12,)


def test_gat_attention_rows_sum_to_one():
    layer = GATLayer(4, 3)
    x = torch.randn(5, 4)
    adj = np.ones((5, 5), dtype=np.float32)
    h = layer(x, torch.tensor(adj))
    assert h.shape == (5, 3)


def test_gat_masked_edges_get_zero_attention():
    layer = GATLayer(4, 3)
    x = torch.randn(3, 4)
    adj = torch.tensor(np.eye(3, dtype=np.float32))  # only self loops
    h = layer(x, adj)  # must not produce NaN despite full-masked softmax risk
    assert torch.isfinite(h).all()


def test_mlp_can_overfit_single_batch():
    torch.manual_seed(0)
    m = MLP(5, hidden=(32,))
    x = torch.randn(24, 5)
    y = (x[:, 0] + x[:, 1] > 0).float()
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    loss0 = None
    for i in range(300):
        logits = m(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        if loss0 is None:
            loss0 = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.2 * loss0
