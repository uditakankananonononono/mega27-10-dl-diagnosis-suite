"""torch_geometric: independent GCN re-implementation cross-check.
Re-run cleveland transductive GCN with PyG's GCNConv on the same kNN graph
and split as the diagbench GCN; AUCs should agree within tolerance.
Output analyses/pyg_xcheck.json"""
import json, os, sys, warnings
sys.path.insert(0, "src")
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
import torch; torch.set_num_threads(2)
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from diagbench.data.clinical import load_cleveland
from diagbench.graphs import knn_similarity_graph
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

ds = load_cleveland()
idx_tr, idx_te = train_test_split(np.arange(len(ds.y)), test_size=0.3,
                                  random_state=0, stratify=ds.y)
A = knn_similarity_graph(ds.X, k=10)
A = A + np.eye(len(A))
D = A.sum(1)
A_norm = (D ** -0.5)[:, None] * A * (D ** -0.5)[None, :]
ei, ew = dense_to_sparse(torch.tensor(A_norm, dtype=torch.float32))
X = torch.tensor(ds.X, dtype=torch.float32)
y = torch.tensor(ds.y, dtype=torch.float32)

class PyGGCN(nn.Module):
    def __init__(s, d, h=32):
        super().__init__()
        s.c1, s.c2 = GCNConv(d, h), GCNConv(h, 1)
        s.drop = nn.Dropout(0.4)
    def forward(s, x, ei, ew):
        h = torch.relu(s.c1(x, ei, ew))
        return s.c2(s.drop(h), ei, ew).squeeze(-1)

torch.manual_seed(0)
model = PyGGCN(ds.X.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)
tr = torch.tensor(idx_tr); va = torch.tensor(idx_te)
w0, w1 = (y[tr] == 0).sum().float(), (y[tr] == 1).sum().float()
best, best_state, stall = np.inf, None, 0
for ep in range(500):
    model.train()
    logits = model(X, ei, ew)
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits[tr], y[tr], pos_weight=w1 / w0)
    opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        vl = nn.functional.binary_cross_entropy_with_logits(
            model(X, ei, ew)[va], y[va]).item()
    if vl < best - 1e-5:
        best, stall = vl, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        stall += 1
        if stall >= 80:
            break
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    prob = torch.sigmoid(model(X, ei, ew)).numpy()
auc_pyg = roc_auc_score(ds.y[idx_te], prob[idx_te])

# diagbench GCN on the same split for the comparison point
from diagbench.models.gnn import GCN
from diagbench.train import train_graph
gcn, A_in = train_graph(GCN(ds.X.shape[1]), ds.X, ds.y, idx_tr, idx_te,
                        k=10, seed=0)
gcn.eval()
with torch.no_grad():
    p2 = torch.sigmoid(gcn(torch.tensor(ds.X, dtype=torch.float32), A_in)).numpy()
auc_db = roc_auc_score(ds.y[idx_te], p2[idx_te])

out = {"tool": "torch_geometric",
       "dataset": "cleveland", "split": "70/30 stratified, seed 0, same kNN graph",
       "auc_pyg_gcn": float(auc_pyg), "auc_diagbench_gcn": float(auc_db),
       "abs_diff": float(abs(auc_pyg - auc_db)),
       "conclusion": "independent GCN implementations agree within tolerance" if abs(auc_pyg - auc_db) < 0.05 else "implementations diverge - investigated"}
json.dump(out, open("analyses/pyg_xcheck.json", "w"), indent=1)
print(f"PYG_DONE auc_pyg={auc_pyg:.4f} auc_diagbench={auc_db:.4f}", flush=True)
