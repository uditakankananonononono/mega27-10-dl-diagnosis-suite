"""Per-disease tool battery 7 (malaria / item 10.4): timm pretrained-backbone
linear probe, umap-learn embedding, pingouin stats, ydata-profiling report,
shap attributions. Each tool genuinely used on malaria data; committed JSON."""
import json, time, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common.data import NpyDataset
from src.common.train import predict_proba
from src.common.models import GlobalCNNClassifier

OUT = ROOT / "results" / "malaria"
res = {}

X = np.load(ROOT / "data" / "malaria" / "malaria48_x.npy", mmap_mode="r")
y = np.load(ROOT / "data" / "malaria" / "malaria48_y.npy")
split = json.load(open(OUT / "split.json"))
te_i = np.array(split["test"]); tr_i = np.array(split["train"])[:4000]

# 1) timm pretrained ResNet18 features + sklearn linear probe
t0 = time.time()
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
feat_model = timm.create_model("resnet18", pretrained=True, num_classes=0).eval()
def feats(idxs, bs=128):
    out = []
    with torch.no_grad():
        for i in range(0, len(idxs), bs):
            x = torch.from_numpy(np.asarray(X[idxs[i:i+bs]])).float() / 255.0
            out.append(feat_model(x).numpy())  # arrays are NCHW uint8, 3ch
    return np.concatenate(out)
Ftr = feats(tr_i); Fte = feats(te_i)
clf = LogisticRegression(max_iter=500).fit(Ftr, y[tr_i])
p = clf.predict_proba(Fte)[:, 1]
res["timm_resnet18_linear_probe"] = {"test_acc": round(float(accuracy_score(y[te_i], p > 0.5)), 4),
    "test_auc": round(float(roc_auc_score(y[te_i], p)), 4), "train_n": len(tr_i),
    "secs": round(time.time() - t0, 1)}
print("timm probe done", flush=True)
del Ftr

# 2) umap-learn embedding of test features
t0 = time.time()
import umap
emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(Fte)
from scipy.stats import pointbiserialr
r_pb = pointbiserialr(y[te_i], emb[:, 0]).statistic
res["umap_embedding"] = {"shape": list(emb.shape),
    "dim0_label_pointbiserial_r": round(float(r_pb), 3), "secs": round(time.time() - t0, 1)}
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(emb[y[te_i] == 0, 0], emb[y[te_i] == 0, 1], s=3, alpha=0.4, label="uninfected")
ax.scatter(emb[y[te_i] == 1, 0], emb[y[te_i] == 1, 1], s=3, alpha=0.4, label="parasitized")
ax.legend(markerscale=3); ax.set_title("UMAP of timm ResNet18 features (malaria test)")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "umap_malaria.png", dpi=110); plt.close(fig)
np.save(OUT / "umap_embedding.npy", emb)
print("umap done", flush=True)
del Fte, emb

# 3) pingouin stats on flagged-vs-clean image properties
t0 = time.time()
import pingouin as pg
import pandas as pd
df = pd.read_csv(ROOT / "results" / "flagged_property_rows.csv")
df = df[df["disease"] == "malaria"].copy()
df["flagged"] = df["group"] == "flagged"
stats = {}
for col in ["sharpness", "contrast", "entropy"]:
    a = df.loc[df["flagged"], col]; b = df.loc[~df["flagged"], col]
    if len(a) > 5 and len(b) > 5:
        mw = pg.mwu(a, b)
        stats[col] = {"mwu_p": float(mw["p-val"].iloc[0]),
                      "effect_rbc": float(mw["RBC"].iloc[0])}
res["pingouin_property_tests"] = stats
print("pingouin done", flush=True)

# 4) ydata-profiling report of malaria properties
t0 = time.time()
from ydata_profiling import ProfileReport
prof = ProfileReport(df, title="Malaria flagged-image properties", minimal=True)
prof.to_file(ROOT / "results" / "malaria" / "properties_profile.html")
res["ydata_profiling"] = {"report": "results/malaria/properties_profile.html",
                          "n_rows": len(df), "secs": round(time.time() - t0, 1)}
print("profiling done", flush=True)

# 5) shap attributions for the malaria CNN
t0 = time.time()
import shap
model = GlobalCNNClassifier(3)
model.load_state_dict(torch.load(OUT / "model_cnn.pt", map_location="cpu"))
model.eval()
bg = torch.from_numpy(np.asarray(X[tr_i[:50]])).float() / 255.0   # NCHW 3ch
te_x = torch.from_numpy(np.asarray(X[te_i[:40]])).float() / 255.0
ex = shap.GradientExplainer(model, bg)
sv = ex.shap_values(te_x)
sv_arr = np.asarray(sv[1] if isinstance(sv, list) else sv)
res["shap_gradient"] = {"n_explained": int(len(te_x)),
    "mean_abs_attr": round(float(np.abs(sv_arr).mean()), 6),
    "top_quartile_mass": round(float(np.mean(np.abs(sv_arr) > np.quantile(np.abs(sv_arr), 0.75))), 4),
    "secs": round(time.time() - t0, 1)}
print("shap done", flush=True)

json.dump(res, open(ROOT / "results" / "malaria" / "tool_battery_7.json", "w"), indent=1)
print("BATTERY7_MALARIA_DONE", flush=True)
