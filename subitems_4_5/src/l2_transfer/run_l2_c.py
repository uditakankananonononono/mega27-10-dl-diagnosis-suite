#!/usr/bin/env python3
"""PREREG_PNEUMONIA_L2 Model C: progressive fine-tune of XRV densenet121-res224-all.
Phases: 2 = denseblock4+norm5+classifier, 3 = full (small lr). 3 seeds.
Threshold chosen on VAL only then frozen; single test evaluation per seed.
Batch-level checkpointing: each call processes --batches batches then saves and exits.
Usage: run_l2_c.py train --seed 11 --phase 2 --batches 24
       run_l2_c.py eval  --seed 11          (after all phases trained)
       run_l2_c.py status"""
import argparse, json, time
import numpy as np, torch, torchxrayvision as xrv
torch.set_num_threads(2)
from pathlib import Path
from PIL import Image
from torchvision import transforms

ROOT = Path.home() / "work/repos/mega27-10-dl-diagnosis-suite/subitems_4_5"
DATA = Path.home() / "work/data/kermany_cxr/chest_xray"
OUT = ROOT / "results/l2_c"; OUT.mkdir(parents=True, exist_ok=True)
BATCH = 16
EPOCHS_PER_PHASE = 2
LR = {2: 1e-4, 3: 1e-5}

tf = transforms.Compose([
    transforms.Grayscale(1), transforms.Resize((224, 224)), transforms.ToTensor(),
])

def list_split(split):
    paths, labels = [], []
    for cls, y in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        for p in sorted((DATA / split / cls).glob("*.jpeg")):
            paths.append(str(p)); labels.append(y)
    return paths, np.array(labels, dtype=np.float32)

def load_batch(paths, idx):
    ims = np.stack([xrv.datasets.normalize(tf(Image.open(paths[i]).convert("L")).numpy()[0] * 255.0, 255.0) for i in idx])
    return torch.from_numpy(ims.astype(np.float32))[:, None, :, :]

def fresh_model():
    m = xrv.models.DenseNet(weights="densenet121-res224-all")
    m.classifier = torch.nn.Linear(1024, 1)
    m.op_threshs = None  # single-logit head; disable 18-pathology operating-point norm
    return m

def set_trainable(m, phase):
    for p in m.parameters():
        p.requires_grad = False
    if phase >= 2:
        for name in ["denseblock4", "norm5"]:
            mod = getattr(m.features, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = True
    if phase >= 3:
        for p in m.features.parameters():
            p.requires_grad = True
    for p in m.classifier.parameters():
        p.requires_grad = True

def ckpt_path(seed, phase):
    return OUT / f"ckpt_s{seed}_p{phase}.pt"

def train(seed, phase, max_batches):
    paths, y = list_split("train")
    n = len(paths); n_batches = (n + BATCH - 1) // BATCH
    cp = ckpt_path(seed, phase)
    m = fresh_model()
    if cp.exists():
        st = torch.load(cp, weights_only=False)
        m.load_state_dict(st["model"]); start_ep, start_b = st["epoch"], st["batch"]
        if st.get("phase_done"):
            print("PHASE ALREADY DONE"); return
    elif phase > 2 and ckpt_path(seed, phase - 1).exists():
        st = torch.load(ckpt_path(seed, phase - 1), weights_only=False)
        m.load_state_dict(st["model"]); start_ep, start_b = 0, 0
        print(f"initialized phase {phase} from phase {phase-1} weights")
    else:
        start_ep, start_b = 0, 0
    set_trainable(m, phase)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR[phase])
    if cp.exists() and "opt" in st:
        try: opt.load_state_dict(st["opt"])
        except Exception: pass
    m.train()
    torch.manual_seed(seed * 100 + phase)
    done = 0
    ep, b = start_ep, start_b
    t0 = time.time()
    while ep < EPOCHS_PER_PHASE:
        g = torch.Generator().manual_seed(seed * 1000 + phase * 10 + ep)
        order = torch.randperm(n, generator=g).tolist()
        while b < n_batches:
            idx = order[b * BATCH:(b + 1) * BATCH]
            x = load_batch(paths, idx)
            yb = torch.from_numpy(y[idx])
            logit = m(x).squeeze(1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            b += 1; done += 1
            if done % 5 == 0:
                print(f"s{seed} p{phase} ep{ep} b{b}/{n_batches} loss={loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
            if done >= max_batches:
                torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), epoch=ep, batch=b, phase_done=False), cp)
                print(f"CHECKPOINT saved s{seed} p{phase} ep{ep} b{b}", flush=True); return
        ep += 1; b = 0
    torch.save(dict(model=m.state_dict(), epoch=EPOCHS_PER_PHASE, batch=0, phase_done=True), cp)
    print(f"PHASE {phase} DONE seed {seed}", flush=True)

@torch.no_grad()
def predict_split(m, split):
    m.eval()
    paths, y = list_split(split)
    probs = []
    for i in range(0, len(paths), BATCH):
        x = load_batch(paths, list(range(i, min(i + BATCH, len(paths)))))
        probs.append(torch.sigmoid(m(x).squeeze(1)).numpy())
    return np.concatenate(probs), y

def evaluate(seed):
    from sklearn.metrics import roc_auc_score, brier_score_loss
    m = fresh_model()
    st = torch.load(ckpt_path(seed, 3), weights_only=False)
    m.load_state_dict(st["model"])
    pv, yva = predict_split(m, "val")
    ths = np.linspace(0.05, 0.95, 181)
    th = max(ths, key=lambda t: ((pv >= t) == yva).mean())
    pt, yte = predict_split(m, "test")
    acc = float(((pt >= th) == yte).mean())
    auc = float(roc_auc_score(yte, pt)); br = float(brier_score_loss(yte, pt))
    r = dict(seed=int(seed), val_thr=float(th), test_acc=acc, test_auc=auc, test_brier=br)
    f = OUT / "l2_c_results.json"
    res = json.loads(f.read_text()) if f.exists() else dict(model="C", protocol="PREREG_PNEUMONIA_L2", results=[])
    res["results"] = [x for x in res["results"] if x["seed"] != seed] + [r]
    if len(res["results"]) == 3:
        accs = [x["test_acc"] for x in res["results"]]
        res["mean_acc"] = float(np.mean(accs)); res["sd_acc"] = float(np.std(accs, ddof=1))
    f.write_text(json.dumps(res, indent=2))
    print("EVAL", r, flush=True)

def status():
    for seed in [11, 23, 37]:
        for phase in [2, 3]:
            cp = ckpt_path(seed, phase)
            if cp.exists():
                st = torch.load(cp, weights_only=False)
                print(f"s{seed} p{phase}: ep{st['epoch']} b{st['batch']} done={st.get('phase_done')}")
            else:
                print(f"s{seed} p{phase}: not started")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "status"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--phase", type=int, default=2)
    ap.add_argument("--batches", type=int, default=24)
    a = ap.parse_args()
    if a.mode == "train": train(a.seed, a.phase, a.batches)
    elif a.mode == "eval": evaluate(a.seed)
    else: status()
