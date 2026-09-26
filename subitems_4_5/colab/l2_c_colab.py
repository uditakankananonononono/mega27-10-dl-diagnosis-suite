# MEGA27-10 subitems-4_5 - Pneumonia L2 Model C - LOCKED PREREG_PNEUMONIA_L2 protocol, UNCHANGED.
# Colab: Runtime > Change runtime type > T4 GPU. Run this whole cell. Paste back everything between
# RESULTS_JSON_BEGIN and RESULTS_JSON_END. Any assertion failure = abort, fall back to local grind.
import subprocess, sys
subprocess.run([sys.executable,'-m','pip','install','-q','torchxrayvision','scikit-learn'],check=True)
from huggingface_hub import snapshot_download
import numpy as np, torch, torchxrayvision as xrv, json, time
from pathlib import Path
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, brier_score_loss

root = Path(snapshot_download(repo_id="ahulikal/chest-xray-pneumonia-mirror", repo_type="dataset"))
cands = [p for p in root.rglob("chest_xray") if p.is_dir()]
DATA = cands[0] if cands else root
exp = {("train","NORMAL"):1341,("train","PNEUMONIA"):3875,("val","NORMAL"):8,("val","PNEUMONIA"):8,("test","NORMAL"):234,("test","PNEUMONIA"):390}
counts = {(s,c): len(list((DATA/s/c).glob("*.jpeg"))) for s in ["train","val","test"] for c in ["NORMAL","PNEUMONIA"]}
assert counts == exp, f"DATA MISMATCH vs locked mirror counts: {counts}"
print("data OK:", DATA)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
assert DEV == "cuda", "NO GPU - abort per instructions (fall back to local grind)"

BATCH, EPOCHS_PER_PHASE = 16, 2
LR = {2: 1e-4, 3: 1e-5}
tf = transforms.Compose([transforms.Grayscale(1), transforms.Resize((224,224)), transforms.ToTensor()])

def list_split(split):
    paths, labels = [], []
    for cls, y in [("NORMAL",0),("PNEUMONIA",1)]:
        for p in sorted((DATA/split/cls).glob("*.jpeg")):
            paths.append(str(p)); labels.append(y)
    return paths, np.array(labels, dtype=np.float32)

def load_batch(paths, idx):
    ims = np.stack([xrv.datasets.normalize(tf(Image.open(paths[i]).convert("L")).numpy()[0]*255.0, 255.0) for i in idx])
    return torch.from_numpy(ims.astype(np.float32))[:, None, :, :].to(DEV)

def fresh_model():
    m = xrv.models.DenseNet(weights="densenet121-res224-all")
    m.classifier = torch.nn.Linear(1024, 1)
    m.op_threshs = None
    return m.to(DEV)

def set_trainable(m, phase):
    for p in m.parameters(): p.requires_grad = False
    if phase >= 2:
        for name in ["denseblock4","norm5"]:
            mod = getattr(m.features, name, None)
            if mod is not None:
                for p in mod.parameters(): p.requires_grad = True
    if phase >= 3:
        for p in m.features.parameters(): p.requires_grad = True
    for p in m.classifier.parameters(): p.requires_grad = True

def train_phase(m, seed, phase, paths, y):
    set_trainable(m, phase)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR[phase])
    n = len(paths); nb = (n + BATCH - 1)//BATCH
    m.train()
    for ep in range(EPOCHS_PER_PHASE):
        g = torch.Generator().manual_seed(seed*1000 + phase*10 + ep)
        order = torch.randperm(n, generator=g).tolist()
        for b in range(nb):
            idx = order[b*BATCH:(b+1)*BATCH]
            logit = m(load_batch(paths, idx)).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[idx]).to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"s{seed} p{phase} ep{ep} loss={loss.item():.4f}", flush=True)

@torch.no_grad()
def predict_split(m, split):
    m.eval()
    paths, y = list_split(split)
    probs = []
    for i in range(0, len(paths), BATCH):
        probs.append(torch.sigmoid(m(load_batch(paths, list(range(i, min(i+BATCH, len(paths)))))).squeeze(1)).cpu().numpy())
    return np.concatenate(probs), y

t0 = time.time()
paths_tr, y_tr = list_split("train")
results = []
for seed in [11, 23, 37]:
    torch.manual_seed(seed)
    m = fresh_model()
    train_phase(m, seed, 2, paths_tr, y_tr)
    train_phase(m, seed, 3, paths_tr, y_tr)
    pv, yva = predict_split(m, "val")
    ths = np.linspace(0.05, 0.95, 181)
    th = max(ths, key=lambda t: ((pv >= t) == yva).mean())
    pt, yte = predict_split(m, "test")
    r = dict(seed=int(seed), val_thr=float(th),
             test_acc=float(((pt >= th) == yte).mean()),
             test_auc=float(roc_auc_score(yte, pt)),
             test_brier=float(brier_score_loss(yte, pt)))
    results.append(r); print("SEED", r, flush=True)
accs = [r["test_acc"] for r in results]
res = dict(model="C", protocol="PREREG_PNEUMONIA_L2", runner="colab-gpu", results=results,
           mean_acc=float(np.mean(accs)), sd_acc=float(np.std(accs, ddof=1)), wall_s=time.time()-t0)
print("RESULTS_JSON_BEGIN"); print(json.dumps(res, indent=2)); print("RESULTS_JSON_END")
