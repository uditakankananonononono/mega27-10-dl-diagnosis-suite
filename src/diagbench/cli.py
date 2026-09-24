"""CLI: python -m diagbench.cli run --dataset wdbc [--models cnn1d gcn]"""
from __future__ import annotations

import argparse
import json

from .benchmark import DEEP_MODELS, SKLEARN_BASELINES, run_benchmark, save_results
from .data.clinical import load_dataset


def main(argv=None):
    ap = argparse.ArgumentParser(prog="diagbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--dataset", required=True)
    run.add_argument("--models", nargs="*",
                     default=list(DEEP_MODELS) + list(SKLEARN_BASELINES))
    run.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    run.add_argument("--k", type=int, default=10)
    run.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "run":
        ds = load_dataset(args.dataset)
        result = run_benchmark(ds, models=tuple(args.models),
                               seeds=tuple(args.seeds), k=args.k)
        out = args.out or f"results/{args.dataset}.json"
        save_results(result, out)
        for name, block in result["models"].items():
            s = block["summary"]
            print(f"{name:14s} AUC {s['roc_auc']['mean']:.3f}+-{s['roc_auc']['std']:.3f}"
                  f"  ACC {s['accuracy']['mean']:.3f}  BRIER {s['brier']['mean']:.3f}"
                  f"  ECE {s['ece']['mean']:.3f}")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
