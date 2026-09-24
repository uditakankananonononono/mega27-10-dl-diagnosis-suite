# diagbench - Deep-Learning Diagnosis Suite (MEGA-27 Item 10)

A tested, reproducible toolkit for benchmarking deep-learning diagnosis
models on real clinical datasets, with graph-neural-network patient models,
a cross-disease discovery experiment, and a label-efficiency benchmark.

## Install

    pip install -e .        # or: PYTHONPATH=src python -m diagbench.cli ...

## Quick start

    # full benchmark on one dataset (6 models x 5 seeds, bootstrap CIs)
    python -m diagbench.cli run --dataset wdbc
    python -m diagbench.cli run --dataset cleveland --models mlp gcn

## What it ships

- **26 accession-level clinical datasets** (UCI/NIDDK, URLs verified):
  loaders with documented label coding, median imputation, factorized
  categoricals. `diagbench.data.clinical` (core 4) and
  `diagbench.data.panel` (extended 22).
- **Models**: MLP, 1D-CNN, GCN, GAT over kNN patient-similarity graphs
  (self-tuned Gaussian kernel, memory-safe matmul distances), plus
  scikit-learn logistic regression / random forest baselines.
- **Honest evaluation**: ROC AUC (Mann-Whitney), balanced accuracy, Brier,
  ECE, percentile bootstrap 95% CIs, 5 seeded splits.
- **Discovery experiments** (`diagbench.xgraph`): a unified cross-disease
  patient graph with disease-agnostic distributional fingerprints; paired
  bridge ablation with exact sign-test falsification; stable-bridge mining.
- **Label-efficiency benchmark**: transductive GCN at 10-100% labels vs
  full-label tabular models.
- **30 hermetic tests** (no network): `python -m pytest`.

## Repository layout

    src/diagbench/     package (data, models, graphs, train, eval, xgraph, cli)
    tests/             hermetic pytest suite
    results/           every number in the paper, as committed JSON
    scripts/           paper generator (Times New Roman DOCX)
    paper/             LaTeX source of record + generated DOCX/PDF
    run_*.py           experiment runners (benchmark, panel, xgraph,
                       label-efficiency, MedMNIST arm)

All results regenerate end-to-end from the runners.
