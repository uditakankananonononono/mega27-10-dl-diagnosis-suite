#!/usr/bin/env python3
"""Benchmark Integrity Scanner v0 - generalizes this repo's label-census pipeline.

Judge round 3 novelty change (isef_judge/pneumonia_round3.md): the reusable artifact is a
dataset-agnostic audit that asks "should this benchmark's labels be trusted?" before asking
"can AI beat this benchmark?".

Contract (v1 target):
  INPUT : dataset dir (images/) + labels.csv + a probe-model spec
  OUTPUT: integrity_report.json with the fixed schema below

v0 (this file): assembles the report from ALREADY-COMPUTED census artifacts of two real
datasets (Kermany pneumonia: results/pneumonia/flag_tiers.json; NIH malaria:
results/malaria/census_crosscheck.json), proving the schema on real data before the
end-to-end probe retrain is wired in. No numbers are recomputed here - every figure is read
from the committed artifacts and cited by path.
"""
import json, sys, pathlib

def pneumonia_report(root):
    d = json.load(open(root / "results/pneumonia/flag_tiers.json"))
    return {
        "dataset": "kermany_cxr_pneumonia",
        "source_artifact": "results/pneumonia/flag_tiers.json",
        "estimated_inconsistent_labels": d["protocol"],
        "review_tiers": {
            "high_confidence": d["tier1_core_count"],
            "medium_confidence": d["tier2_canonical_only_count"] + d["tier2_rerun_only_count"],
        },
        "validation_status": "computational candidates only; blinded expert adjudication NOT yet done (judge weakness #1, open)",
    }

def malaria_report(root):
    d = json.load(open(root / "results/malaria/census_crosscheck.json"))
    return {
        "dataset": "nih_malaria_cells",
        "source_artifact": "results/malaria/census_crosscheck.json",
        "cross_method_agreement": {
            "ours_flagged": d["ours_flagged"] if isinstance(d["ours_flagged"], int) else len(d["ours_flagged"]),
            "cleanlab_flagged": d["cleanlab_flagged"] if isinstance(d["cleanlab_flagged"], int) else len(d["cleanlab_flagged"]),
            "intersection": d["intersection"] if isinstance(d["intersection"], int) else len(d["intersection"]),
            "jaccard": d["jaccard"],
        },
        "agreed_ids": d["agreed_ids"] if isinstance(d["agreed_ids"], int) else len(d["agreed_ids"]),
        "validation_status": "computational candidates only",
    }

def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    report = {
        "scanner_version": "0.1",
        "question": "Should this benchmark's labels be trusted before they judge diagnostic AI?",
        "datasets": [pneumonia_report(root), malaria_report(root)],
        "planned_demos": ["MedMNIST panels (free)", "one non-medical image benchmark"],
    }
    out = root / "results/integrity_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:600])

if __name__ == "__main__":
    sys.exit(main())
