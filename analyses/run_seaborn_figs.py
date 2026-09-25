"""seaborn publication figures over the committed result panels.
Output figures/panel_auc_by_arm.png, figures/label_eff_curves.png,
analyses/seaborn_figs.json"""
import json, os, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", font="serif")

geo = json.load(open("results/geo_panel.json"))
om = json.load(open("results/openml_panel.json"))
xe = json.load(open("results/xena_panel.json"))
mm = json.load(open("results/medmnist.json"))
m3 = json.load(open("results/medmnist3d.json"))
rows = []
for k, v in geo.items(): rows.append(("GEO", v["auc"]))
for k, v in om.items(): rows.append(("OpenML", v["best_auc"]))
for k, v in xe.items(): rows.append(("TCGA (Xena)", v["auc"]))
for k, v in mm.items(): rows.append(("MedMNIST 2D", v["auc"]))
for k, v in m3.items(): rows.append(("MedMNIST 3D", v["auc"]))
import pandas as pd
df = pd.DataFrame(rows, columns=["arm", "auc"])
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.violinplot(data=df, x="arm", y="auc", inner=None, cut=0, ax=ax,
               color="0.85")
sns.stripplot(data=df, x="arm", y="auc", size=4, jitter=0.22, ax=ax,
              color="0.15", alpha=0.7)
ax.axhline(0.5, ls="--", lw=0.8, c="0.4")
ax.set_ylabel("cross-validated ROC AUC"); ax.set_xlabel("")
ax.set_title("AUC distribution across the five dataset arms (125 datasets)")
fig.tight_layout(); fig.savefig("figures/panel_auc_by_arm.png", dpi=150)

le = json.load(open("results/label_efficiency.json"))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for name in le:
    fr = [float(f) for f in ("0.1", "0.25", "0.5", "1.0") if f in le[name]["gcn"]]
    g = [np.mean(le[name]["gcn"][f"{f}"]) for f in fr]
    m = [np.mean(le[name]["mlp"][f"{f}"]) for f in fr]
    ax.plot([f * 100 for f in fr], g, "o-", label=f"{name} GCN")
    ax.plot([f * 100 for f in fr], m, "s--", label=f"{name} MLP")
ax.set_xlabel("% of labels used"); ax.set_ylabel("ROC AUC")
ax.set_title("Label efficiency: GCN vs MLP")
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("figures/label_eff_curves.png", dpi=150)

out = {"tool": "seaborn",
       "figures": ["figures/panel_auc_by_arm.png", "figures/label_eff_curves.png"],
       "arm_medians": df.groupby("arm")["auc"].median().round(4).to_dict(),
       "n_datasets": int(len(df))}
json.dump(out, open("analyses/seaborn_figs.json", "w"), indent=1)
print("SEABORN_DONE", flush=True)
