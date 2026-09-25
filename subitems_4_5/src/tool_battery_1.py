"""Tool battery 1: four independent small analyses, each answering a real
question about the study. All results committed as JSON/PNG."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path(__file__).resolve().parent.parent
out = {}

# --- 1. sympy: verify the GCN normalisation formula used in models.py ---
# claim in Methods: A_hat = D^{-1/2}(A+I)D^{-1/2} is symmetric with spectrum in (-1,1]
import sympy as sp
from src.common.models import grid_adjacency, gcn_norm
for g in (2, 3):
    A = sp.Matrix(grid_adjacency(g).numpy().tolist())
    I = sp.eye(g * g)
    D = sp.diag(*[1 / sp.sqrt(sum(A.row(i)) + 1) for i in range(g * g)])
    Ah = D * (A + I) * D
    ev = [complex(v) for v in Ah.eigenvals().keys()]
    ours = gcn_norm(grid_adjacency(g)).numpy()
    out[f"sympy_gcn_norm_grid{g}"] = {
        "symbolic_matches_numpy": bool(np.allclose(np.array(Ah.tolist(), dtype=float), ours, atol=1e-10)),
        "symmetric": bool((Ah - Ah.T) == sp.zeros(g * g)),
        "spectral_range": [float(min(e.real for e in ev)), float(max(e.real for e in ev))],
        "spectrum_in_bound": all(-1 - 1e-9 <= e.real <= 1 + 1e-9 for e in ev)}

# --- 2. networkx: region-grid graph properties (mixing claims in Methods) ---
import networkx as nx
for g in (3, 4):
    G = nx.from_numpy_array(grid_adjacency(g).numpy())
    out[f"networkx_grid{g}"] = {
        "n_nodes": G.number_of_nodes(), "diameter": nx.diameter(G),
        "algebraic_connectivity": round(nx.algebraic_connectivity(G), 6),
        "avg_clustering": round(nx.average_clustering(G), 6),
        "two_layer_full_mixing": bool(nx.diameter(G) <= 2),
        "mixing_note": ("diameter 2: two GCN layers mix all region pairs" if nx.diameter(G) <= 2
                        else f"diameter {nx.diameter(G)}: two GCN layers leave farthest region pairs unmixed (grid=4 limitation)")}

# --- 3. cv2: does the resize implementation move model confidence? ---
import cv2, torch
from PIL import Image
from src.common.data import NpyDataset
from src.common.models import GlobalCNNClassifier
m = GlobalCNNClassifier(3)
m.load_state_dict(torch.load(ROOT / "results/malaria/model_cnn.pt", map_location="cpu"))
m.eval()
split = json.load(open(ROOT / "results/malaria/split.json"))
import glob
pngs = sorted(glob.glob(str(ROOT / "data/malaria/cell_images/*/*.png")))
rng = np.random.default_rng(1)
idx = rng.choice(len(pngs), 200, replace=False)
diffs = []
for i in idx:
    arr = np.asarray(Image.open(pngs[int(i)]).convert("RGB"))   # native ~110-160px
    a = cv2.resize(arr, (48, 48), interpolation=cv2.INTER_AREA)
    b = cv2.resize(arr, (48, 48), interpolation=cv2.INTER_LINEAR)
    xa = torch.from_numpy(a.transpose(2, 0, 1)[None].astype(np.float32) / 255)
    xb = torch.from_numpy(b.transpose(2, 0, 1)[None].astype(np.float32) / 255)
    with torch.no_grad():
        pa = torch.softmax(m(xa), 1)[0, 1].item(); pb = torch.softmax(m(xb), 1)[0, 1].item()
    diffs.append(abs(pa - pb))
out["cv2_resize_robustness"] = {
    "n": 200, "max_abs_prob_delta": round(float(np.max(diffs)), 6),
    "mean_abs_prob_delta": round(float(np.mean(diffs)), 6),
    "note": "native-resolution PNGs downsized to 48px with two cv2 interpolations; answers whether resizer choice moves committed-model confidence"}

# --- 4. seaborn: OOF probability distributions showing one-directional tails ---
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
for disease in ("malaria", "pneumonia"):
    oof = json.load(open(ROOT / f"results/{disease}/oof_probs_train.json"))
    split_d = json.load(open(ROOT / f"results/{disease}/split.json"))
    probs = np.array(oof["probs"])
    ds_d = NpyDataset(str(ROOT / "data" / disease / ("malaria48" if disease == "malaria" else "cxr_train")), train=False)
    y = np.asarray(ds_d.y)[np.array(split_d["train"])]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for c, name in enumerate(ds_d.classes):
        sns.kdeplot(probs[y == c, 1], ax=ax, label=f"true {name}", fill=True, alpha=0.35)
    ax.set_xlabel("OOF P(positive class)"); ax.set_yticks([]); ax.legend()
    ax.set_title(f"{disease}: out-of-fold confidence by true label")
    fig.tight_layout(); fig.savefig(ROOT / "figures" / f"{disease}_oof_distributions.png", dpi=150)
    plt.close(fig)
    out[f"seaborn_oof_{disease}"] = {"figure": f"figures/{disease}_oof_distributions.png",
        "tail_mass_wrong_side": {name: round(float((probs[y == c, 1] < 0.1).mean() if c == 1 else (probs[y == c, 1] > 0.9).mean()), 5)
                                  for c, name in enumerate(ds_d.classes)}}

json.dump(out, open(ROOT / "results/tool_battery_1.json", "w"), indent=2, default=str)
print(json.dumps(out, indent=1, default=str)[:2200])
