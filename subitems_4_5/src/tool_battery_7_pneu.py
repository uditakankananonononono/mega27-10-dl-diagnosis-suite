"""Per-disease tool battery 7 (pneumonia / item 10.5): torchxrayvision
pretrained CXR benchmark, kornia stability audit, shap attributions,
pingouin stats, ydata-profiling report. Committed JSON outputs."""
import json, time, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common.models import GlobalCNNClassifier

OUT = ROOT / "results" / "pneumonia"
res = {}

X = np.load(ROOT / "data" / "pneumonia" / "cxr_test_x.npy", mmap_mode="r")
y = np.load(ROOT / "data" / "pneumonia" / "cxr_test_y.npy")
from sklearn.metrics import roc_auc_score, accuracy_score

# 1) torchxrayvision pretrained DenseNet (external benchmark)
t0 = time.time()
import torchxrayvision as xrv
import torch.nn.functional as F
_cached = OUT / "test_probs_torchxrayvision.json"
if _cached.exists():
    _c = json.load(open(_cached))
    probs = np.array(_c["probs"])
    res["torchxrayvision_densenet121_all"] = {
        "test_auc_pneumonia": round(float(roc_auc_score(y, probs)), 4),
        "test_acc_0p5": round(float(accuracy_score(y, probs > 0.5)), 4),
        "n": len(y), "secs": "cached-from-prior-run"}
    print("xrv skipped (cached)", flush=True)
else:
    model_xrv = xrv.models.DenseNet(weights="densenet121-res224-all").eval()
    probs = []
    with torch.no_grad():
        for i in range(len(X)):
            img = np.asarray(X[i, 0]).astype(np.float32)
            img = xrv.datasets.normalize(img, 255)
            img = torch.from_numpy(img)[None, None, ...]
            img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
            o = model_xrv(img)[0]
            probs.append(float(o[model_xrv.pathologies.index("Pneumonia")]))
    probs = np.array(probs)
    res["torchxrayvision_densenet121_all"] = {
        "test_auc_pneumonia": round(float(roc_auc_score(y, probs)), 4),
        "test_acc_0p5": round(float(accuracy_score(y, probs > 0.5)), 4),
        "n": len(y), "secs": round(time.time() - t0, 1)}
    json.dump({"probs": probs.tolist(), "labels": y.tolist()},
              open(OUT / "test_probs_torchxrayvision.json", "w"))
    print("xrv done", flush=True)
    del model_xrv

import gc; gc.collect()
# 2) kornia augmentation-stability audit of the tuned CNN
t0 = time.time()
import kornia.augmentation as K
model = GlobalCNNClassifier(1)
model.load_state_dict(torch.load(OUT / "model_cnn_tuned.pt", map_location="cpu"))
model.eval()
aug = torch.nn.Sequential(K.RandomAffine(degrees=8, translate=(0.05, 0.05), p=1.0),
                          K.RandomBrightness(brightness=(0.0, 0.2), p=1.0))
xs = torch.from_numpy(np.asarray(X[:96])).float() / 255.0
with torch.no_grad():
    p0 = torch.softmax(model(xs), 1)[:, 1]
    torch.manual_seed(0)
    p1 = torch.softmax(model(aug(xs)), 1)[:, 1]
shift = (p1 - p0).abs()
res["kornia_stability"] = {"n": int(len(xs)),
    "mean_abs_prob_shift": round(float(shift.mean()), 4),
    "max_abs_prob_shift": round(float(shift.max()), 4),
    "flip_rate_0p5": round(float(((p0 > 0.5) != (p1 > 0.5)).float().mean()), 4),
    "secs": round(time.time() - t0, 1)}
print("kornia done", flush=True)

# 3) shap attributions for the tuned pneumonia CNN
t0 = time.time()
import shap
bg = torch.from_numpy(np.asarray(X[:16])).float() / 255.0
te_x = torch.from_numpy(np.asarray(X[200:216])).float() / 255.0
ex = shap.GradientExplainer(model, bg)
sv = ex.shap_values(te_x)
sv_arr = np.asarray(sv[1] if isinstance(sv, list) else sv)
res["shap_gradient"] = {"n_explained": int(len(te_x)),
    "mean_abs_attr": round(float(np.abs(sv_arr).mean()), 6),
    "secs": round(time.time() - t0, 1)}
print("shap done", flush=True)
del model, bg, te_x

# 4) pingouin stats on flagged-vs-clean properties
t0 = time.time()
import pingouin as pg
import pandas as pd
df = pd.read_csv(ROOT / "results" / "flagged_property_rows.csv")
df = df[df["disease"] == "pneumonia"].copy()
df["flagged"] = df["group"] == "flagged"
stats = {}
for col in ["sharpness", "contrast", "entropy"]:
    a = df.loc[df["flagged"], col]; b = df.loc[~df["flagged"], col]
    if len(a) > 5 and len(b) > 5:
        mw = pg.mwu(a, b)
        stats[col] = {"mwu_p": float(mw["p_val"].iloc[0]),
                      "effect_rbc": float(mw["RBC"].iloc[0])}
res["pingouin_property_tests"] = stats
print("pingouin done", flush=True)

# 5) ydata-profiling report
t0 = time.time()
from ydata_profiling import ProfileReport
ProfileReport(df, title="Pneumonia flagged-image properties", minimal=True).to_file(
    OUT / "properties_profile.html")
res["ydata_profiling"] = {"report": "results/pneumonia/properties_profile.html",
                          "n_rows": len(df), "secs": round(time.time() - t0, 1)}
print("profiling done", flush=True)

json.dump(res, open(OUT / "tool_battery_7.json", "w"), indent=1)
print("BATTERY7_PNEU_DONE", flush=True)
