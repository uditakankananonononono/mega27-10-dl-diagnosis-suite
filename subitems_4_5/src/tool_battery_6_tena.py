"""Tool battery 6 - item 10a scope extension (suite tool-gate support).
Analyses run over 10a's committed panel results (22 tabular diagnosis
datasets, top-level results/panel_*.json in the same repo family).
Evidence committed under subitems_4_5/results/tena_scope/.

1. duckdb       - SQL: rank 10a panel datasets by cross-model difficulty;
                  which panels stay hard (best model AUC < 0.7)?
2. agate        - independent descriptive statistics per model across panels.
3. csvkit       - csvstat profile of the flattened metrics CSV (third
                  independent computation of the same aggregates).
4. statsmodels  - Benjamini-Hochberg FDR over "best model beats majority
                  class?" Wilcoxon tests per dataset (across seeds).
5. leather      - SVG chart of per-dataset best-model AUC.
"""
import csv, glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TENA = Path("/home/sandbox/mega27-10-dl-diagnosis-suite/results")
OUT = ROOT / "results" / "tena_scope"
OUT.mkdir(parents=True, exist_ok=True)
R = {}


def flatten():
    rows = []
    for f in sorted(glob.glob(str(TENA / "panel_*.json"))):
        j = json.load(open(f))
        ds = j.get("dataset") or Path(f).stem.replace("panel_", "")
        for model, block in (j.get("models") or {}).items():
            for rec in block.get("per_seed", []):
                if rec.get("roc_auc") is None:
                    continue
                rows.append({"dataset": ds, "model": model, "seed_run": len(rows),
                             "accuracy": rec["accuracy"], "roc_auc": rec["roc_auc"],
                             "balanced_accuracy": rec.get("balanced_accuracy"),
                             "f1": rec.get("f1")})
    dest = OUT / "tena_panel_metrics.csv"
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "model", "seed_run",
                                           "accuracy", "roc_auc",
                                           "balanced_accuracy", "f1"])
        w.writeheader()
        w.writerows(rows)
    return rows, dest


def duckdb_analysis(rows):
    import duckdb, pandas as pd
    df = pd.DataFrame(rows)
    con = duckdb.connect()
    con.register("m", df)
    hard = con.execute("""
        SELECT dataset, count(DISTINCT model) AS models, round(max(roc_auc),4) AS best_auc,
               round(avg(roc_auc),4) AS mean_auc
        FROM (SELECT dataset, model, avg(roc_auc) AS roc_auc FROM m
              GROUP BY dataset, model)
        GROUP BY dataset ORDER BY best_auc ASC
    """).df()
    by_model = con.execute("""
        SELECT model, count(DISTINCT dataset) AS datasets,
               round(avg(roc_auc),4) AS mean_auc, round(median(roc_auc),4) AS med_auc
        FROM m GROUP BY model ORDER BY mean_auc DESC
    """).df()
    hard.to_csv(OUT / "tena_dataset_difficulty.csv", index=False)
    by_model.to_csv(OUT / "tena_model_summary.csv", index=False)
    return {"n_hard_below_0.7": int((hard.best_auc < 0.7).sum()),
            "hardest": hard.head(5).to_dict("records"),
            "by_model": by_model.to_dict("records")}


def agate_stats():
    import agate
    tbl = agate.Table.from_csv(str(OUT / "tena_panel_metrics.csv"))
    by_model = tbl.group_by("model")
    summary = by_model.aggregate([
        ("n_runs", agate.Count()),
        ("mean_auc", agate.Mean("roc_auc")),
        ("median_auc", agate.Median("roc_auc")),
        ("min_auc", agate.Min("roc_auc")),
        ("max_auc", agate.Max("roc_auc")),
    ])
    def conv(v):
        from decimal import Decimal
        return round(float(v), 4) if isinstance(v, (float, Decimal)) else v
    out = [{c: conv(r[c]) for c in summary.column_names} for r in summary.rows]
    return {"per_model": out}


def csvkit_profile():
    import subprocess
    r = subprocess.run(["csvstat", "--mean", "--median",
                        str(OUT / "tena_panel_metrics.csv")],
                       capture_output=True, text=True)
    return {"csvstat_excerpt": [ln.strip() for ln in r.stdout.splitlines()
                                if "roc_auc" in ln or "Mean" in ln or "Med" in ln][:8],
            "returncode": r.returncode}


def statsmodels_fdr():
    import numpy as np
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    import pandas as pd
    df = pd.DataFrame(flatten_rows)
    pvals, names = [], []
    for ds, sub in df.groupby("dataset"):
        best = sub.groupby("model")["roc_auc"].mean().idxmax()
        aucs = sub[sub.model == best].roc_auc.values
        maj = sub.groupby("model")["accuracy"].mean()
        # majority-class accuracy proxy: max class share per seed not stored;
        # use "AUC differs from 0.5" as the null per dataset instead
        if len(aucs) >= 3 and np.ptp(aucs) > 0:
            try:
                p = wilcoxon(aucs - 0.5, alternative="greater").pvalue
            except Exception:
                p = 1.0
        else:
            p = 1.0
        pvals.append(float(p)); names.append(ds)
    rej, q, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    out = sorted(zip(names, pvals, q.tolist(), rej.tolist()), key=lambda t: t[1])
    return {"n_datasets": len(names),
            "n_significant_fdr05": int(sum(rej)),
            "most_significant": [{"dataset": n, "p": round(p, 6), "q": round(qq, 6)}
                                 for n, p, qq, rj in out[:5]],
            "null": "per-dataset best-model seed AUCs > 0.5 (one-sided Wilcoxon)"}


def leather_chart():
    import leather, csv as csvmod
    df = list(csvmod.DictReader(open(OUT / "tena_dataset_difficulty.csv")))
    pairs = sorted(((row["dataset"], float(row["best_auc"])) for row in df),
                   key=lambda t: t[1])
    data = [[auc, name] for name, auc in pairs]  # leather Bars: [Number X, Text Y]
    chart = leather.Chart("10a panel: best-model AUC per dataset (ascending)")
    chart.add_bars(data)
    dest = OUT / "tena_panel_auc.svg"
    chart.to_svg(str(dest))
    return {"artifact": str(dest.relative_to(ROOT))}


if __name__ == "__main__":
    flatten_rows, dest = flatten()
    print("rows:", len(flatten_rows), flush=True)
    R["duckdb_difficulty"] = duckdb_analysis(flatten_rows)
    print("duckdb hard panels:", R["duckdb_difficulty"]["n_hard_below_0.7"], flush=True)
    R["agate_stats"] = agate_stats()
    R["csvkit_profile"] = csvkit_profile()
    R["statsmodels_fdr"] = statsmodels_fdr()
    print("fdr significant:", R["statsmodels_fdr"]["n_significant_fdr05"], flush=True)
    R["leather_chart"] = leather_chart()
    json.dump(R, open(OUT / "tool_battery_6_tena.json", "w"), indent=2)
    print("BATTERY6_DONE", flush=True)
