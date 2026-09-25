"""Pneumonia tool battery 8 (item 10.5 depth): SimpleITK resampling audit,
pytorch-grad-cam explanations, scikit-optimize search cross-check.
Committed: results/pneumonia/tool_battery_8.json"""
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

model = GlobalCNNClassifier(1)
model.load_state_dict(torch.load(OUT / "model_cnn_tuned.pt", map_location="cpu"))
model.eval()

# 1) SimpleITK resampling robustness: sitk bilinear vs nearest vs our bilinear
t0 = time.time()
import SimpleITK as sitk
from sklearn.metrics import roc_auc_score
def sitk_resize(arr, interpolator):
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing((1.0, 1.0))
    new = [128, 128]
    rs = sitk.ResampleImageFilter()
    rs.SetSize(new); rs.SetInterpolator(interpolator)
    rs.SetOutputSpacing([128.0 / new[0], 128.0 / new[1]])
    rs.SetOutputOrigin([0.5, 0.5])
    out = rs.Execute(img)
    return sitk.GetArrayFromImage(out)[:128, :128]
sub = np.asarray(X[:200, 0]).astype(np.float32)
lin = np.stack([sitk_resize(a, sitk.sitkLinear) for a in sub])
near = np.stack([sitk_resize(a, sitk.sitkNearestNeighbor) for a in sub])
with torch.no_grad():
    p_lin = torch.softmax(model(torch.from_numpy(lin[:, None] / 255.0)), 1)[:, 1]
    p_near = torch.softmax(model(torch.from_numpy(near[:, None] / 255.0)), 1)[:, 1]
res["simpleitk_resampling"] = {"n": int(len(sub)),
    "auc_sitk_linear": round(float(roc_auc_score(y[:200], p_lin.numpy())), 4),
    "mean_abs_shift_linear_vs_nearest": round(float((p_lin - p_near).abs().mean()), 4),
    "secs": round(time.time() - t0, 1)}
print("simpleitk done", flush=True)
del lin, near

# 2) pytorch-grad-cam on the tuned CNN
t0 = time.time()
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
target_layer = model.core.trunk[-2] if hasattr(model.core, "trunk") else list(model.core.children())[-1]
cam = GradCAM(model=model, target_layers=[target_layer])
te_x = torch.from_numpy(np.asarray(X[200:220])).float() / 255.0
targets = [ClassifierOutputTarget(1)]
grayscale = cam(input_tensor=te_x, targets=targets)
lung_frac = float(np.mean(grayscale.reshape(len(grayscale), -1)[:, :] > 0.5))
res["pytorch_grad_cam"] = {"n_explained": int(len(te_x)),
    "mean_cam_area_over_0p5": round(lung_frac, 4),
    "target_layer": str(target_layer.__class__.__name__),
    "secs": round(time.time() - t0, 1)}
np.save(OUT / "gradcam_test.npy", grayscale)
print("gradcam done", flush=True)
del cam, te_x

# 3) scikit-optimize cross-check of the tuned recipe (gp_minimize over lr, weight decay)
t0 = time.time()
from skopt import gp_minimize
from skopt.space import Real
from src.common.data import NpyDataset
from src.common.train import train_model, predict_proba
from torch.utils.data import Subset
split = json.load(open(OUT / "split.json"))
labels_all = np.load(ROOT / "data" / "pneumonia" / "cxr_train_y.npy")
tr_i = np.array(split["train"])[::4]  # quarter subsample for search speed
va_i = np.array(split["val"])
tr_full = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_train"), train=True, seed=0)
ev = NpyDataset(str(ROOT / "data" / "pneumonia" / "cxr_train"), train=False)
def obj(params):
    lr = params[0]
    torch.manual_seed(0)
    m = GlobalCNNClassifier(1)
    m, hist, best = train_model(m, Subset(tr_full, tr_i), Subset(ev, va_i),
                                epochs=1, batch=32, lr=lr, seed=0)
    probs, yy = predict_proba(m, Subset(ev, va_i))
    return -roc_auc_score(yy, probs[:, 1])
r = gp_minimize(obj, [Real(1e-4, 2e-3, prior="log-uniform", name="lr")],
                n_calls=10, n_initial_points=5, random_state=0)
res["skopt_search"] = {"best_lr": float(r.x[0]),
    "best_val_auc": round(float(-r.fun), 4), "n_calls": 10,
    "note": "1-epoch quarter-subsample search; 5 random + 5 GP-guided calls",
    "secs": round(time.time() - t0, 1)}
print("skopt done", flush=True)

json.dump(res, open(OUT / "tool_battery_8.json", "w"), indent=1)
print("BATTERY8_PNEU_DONE", flush=True)
