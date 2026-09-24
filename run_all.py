import os, sys, time, traceback
sys.path.insert(0, "src")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import torch; torch.set_num_threads(2)
from diagbench.data.clinical import load_dataset
from diagbench.benchmark import DEEP_MODELS, SKLEARN_BASELINES, run_benchmark, save_results
MODELS = tuple(DEEP_MODELS) + tuple(SKLEARN_BASELINES)
for name in ("wdbc", "cleveland", "pima", "parkinsons"):
    t0 = time.time()
    try:
        ds = load_dataset(name)
        res = run_benchmark(ds, models=MODELS, seeds=(0,1,2,3,4), k=10)
        out = f"results/{name}.json"
        save_results(res, out)
        print(f"DONE {name} {time.time()-t0:.0f}s -> {out}", flush=True)
    except Exception:
        print(f"FAIL {name}", flush=True)
        traceback.print_exc()
print("ALL_DONE", flush=True)
