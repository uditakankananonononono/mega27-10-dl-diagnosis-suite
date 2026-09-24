"""Cross-disease unified patient graph: full + bridge-ablated, 5 seeds."""
import json, os, sys, time
sys.path.insert(0, "src")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import torch; torch.set_num_threads(2)
from diagbench.data.base import stratified_split
from diagbench.data.clinical import load_dataset
from diagbench import xgraph
from diagbench.benchmark import _val_split

NAMES = ("wdbc", "cleveland", "pima", "parkinsons")
SEEDS = (0, 1, 2, 3, 4)

datasets = [load_dataset(n) for n in NAMES]
print("loaded", [f"{ds.name}:{ds.X.shape}" for ds in datasets], flush=True)

runs = []
all_seed_edges = []
for seed in SEEDS:
    t0 = time.time()
    itr, iva, ite = [], [], []
    for ds in datasets:
        tr, te = stratified_split(ds.y, 0.25, seed)
        t2, v2 = _val_split(tr, ds.y, seed)
        itr.append(tr); iva.append(v2); ite.append(te)
    encs, heads, Z = xgraph.train_encoders(datasets, itr, iva, seed=seed)
    metas = [xgraph.meta_fingerprint(ds.X) for ds in datasets]
    Zall, A, A_cross, block = xgraph.build_unified_graph(Z, metas, k=10)
    ys = [ds.y for ds in datasets]
    m_full = xgraph.train_multitask_gcn(Zall, A, block, ys, itr, iva, seed=seed)
    res_full = xgraph.evaluate_per_disease(m_full, Zall, A, block, ys, ite)
    A_abl = A * (block[:, None] == block[None, :])
    m_abl = xgraph.train_multitask_gcn(Zall, A_abl, block, ys, itr, iva, seed=seed)
    res_abl = xgraph.evaluate_per_disease(m_abl, Zall, A_abl, block, ys, ite)
    edges = xgraph.bridge_edges(A_cross, block, ys)
    all_seed_edges.append(edges)
    runs.append({"seed": seed,
                 "full": {NAMES[int(d)]: m for d, m in res_full.items()},
                 "ablated": {NAMES[int(d)]: m for d, m in res_abl.items()},
                 "n_cross_edges": int((A_cross > 0).sum() // 2)})
    print(f"seed {seed} done {time.time()-t0:.0f}s "
          f"cross_edges={runs[-1]['n_cross_edges']}", flush=True)

agg = {}
for d_i, name in enumerate(NAMES):
    full = np.array([r["full"][name]["roc_auc"] for r in runs])
    abl = np.array([r["ablated"][name]["roc_auc"] for r in runs])
    acc = np.array([r["full"][name]["accuracy"] for r in runs])
    agg[name] = {
        "full_auc_mean": float(full.mean()), "full_auc_std": float(full.std()),
        "ablated_auc_mean": float(abl.mean()), "ablated_auc_std": float(abl.std()),
        "bridge_delta_auc_mean": float((full - abl).mean()),
        "bridge_delta_auc_per_seed": (full - abl).round(4).tolist(),
        "full_acc_mean": float(acc.mean()),
    }
stable = xgraph.summarize_bridges(all_seed_edges, list(NAMES))
out = {"datasets": {n: {"n": int(len(ds.y)), "d": int(ds.X.shape[1]),
                        "source": ds.source_url, "citation": ds.citation}
                    for n, ds in zip(NAMES, datasets)},
       "k": 10, "seeds": list(SEEDS), "aggregate": agg,
       "stable_bridges": stable, "runs": runs}
json.dump(out, open("results/xgraph.json", "w"), indent=1)
print("XGRAPH_DONE -> results/xgraph.json", flush=True)
for name, a in agg.items():
    print(f"{name:11s} full {a['full_auc_mean']:.4f} vs ablated "
          f"{a['ablated_auc_mean']:.4f}  delta {a['bridge_delta_auc_mean']:+.4f}",
          flush=True)
print("stable bridges:", json.dumps(stable), flush=True)
