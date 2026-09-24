"""Panel benchmark: 22 extended clinical datasets x 4 models x 3 seeds."""
import json, os, sys, time, traceback, warnings
sys.path.insert(0, "src")
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import torch; torch.set_num_threads(2)
from diagbench.data.panel import PANEL_LOADERS
from diagbench.benchmark import run_benchmark, save_results
MODELS = ("mlp", "gcn", "logreg", "random_forest")
for name, fn in PANEL_LOADERS.items():
    out = f"results/panel_{name}.json"
    if os.path.exists(out):
        print(f"SKIP {name}", flush=True); continue
    t0 = time.time()
    try:
        ds = fn()
        res = run_benchmark(ds, models=MODELS, seeds=(0, 1, 2), k=10)
        save_results(res, out)
        best = max(res["models"].items(),
                   key=lambda kv: kv[1]["summary"]["roc_auc"]["mean"])
        print(f"DONE {name} {time.time()-t0:.0f}s best={best[0]} "
              f"{best[1]['summary']['roc_auc']['mean']:.4f}", flush=True)
    except Exception:
        print(f"FAIL {name}", flush=True); traceback.print_exc()
print("PANEL_DONE", flush=True)
