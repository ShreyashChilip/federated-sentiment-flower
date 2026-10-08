"""Figures for the fedbench_v1 screening analysis (post-hoc presentation; no new experiment).

Usage: python make_figures.py <path to results_benchmark/screening>
Reads only the raw run files (final.json, history.json, unseen_clients.json,
final_weights.npz) and tables/per_run_raw.csv, tables/descriptor_correlations.csv.
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SCREEN = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
OUT, T = HERE / "figures", HERE / "tables"
OUT.mkdir(exist_ok=True)
SEEDS = (42, 123, 456)
FL = ["fedavg", "fedprox", "fednova", "scaffold", "fedadam", "fedadagrad", "fedyogi"]
ORDER = ["centralized", "local_only"] + FL
# Validated categorical slots 1-7 for the FL algorithms; neutral ink for the two references.
C = {"fedavg": "#2a78d6", "fedprox": "#eb6834", "fednova": "#1baf7a", "scaffold": "#eda100", "fedadam": "#e87ba4",
     "fedadagrad": "#008300", "fedyogi": "#4a3aa7", "centralized": "#3d3d3a", "local_only": "#8a8980"}
N = dict(centralized="Centralized", local_only="Local-only", fedavg="FedAvg", fedprox="FedProx", fednova="FedNova",
         scaffold="SCAFFOLD", fedadam="FedAdam", fedadagrad="FedAdagrad", fedyogi="FedYogi")
REG = {"yelp_iid": "Yelp, IID (100 clients)", "yelp_dir01": "Yelp, Dirichlet alpha = 0.1 (100 clients)",
       "amazon_vg": "Amazon Video Games, natural product clients"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8980",
                     "axes.labelcolor": "#3d3d3a", "xtick.color": "#3d3d3a", "ytick.color": "#3d3d3a",
                     "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight"})


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(reg, m, s, name):
    return load(SCREEN / reg / m / f"seed{s}" / name)


d = pd.read_csv(T / "per_run_raw.csv")

# Figure 1: held-out macro-F1, dots = seeds, bar = mean, dashed = centralized
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
meths = [m for m in ORDER if m != "local_only"]
for ax, reg in zip(axes, REG):
    for i, m in enumerate(meths):
        v = d[(d.regime == reg) & (d.method == m)].held_f1.values
        ax.plot([v.mean()] * 2, [i - 0.32, i + 0.32], color="#3d3d3a", lw=1.4, zorder=2)
        ax.scatter(v, [i] * len(v), s=26, color=C[m], edgecolors="white", linewidths=0.8, zorder=3)
    ax.axvline(d[(d.regime == reg) & (d.method == "centralized")].held_f1.mean(), color=C["centralized"], ls="--", lw=1)
    ax.set_yticks(range(len(meths)))
    ax.set_yticklabels([N[m] for m in meths])
    ax.set_title(REG[reg], fontsize=8.5, loc="left")
    ax.set_xlabel("Held-out macro-F1" + (" (unseen clients)" if reg == "amazon_vg" else " (official test)"))
fig.suptitle("Figure 1. Final held-out macro-F1 (dots = seeds 42/123/456, bar = mean, dashed = centralized)",
             fontsize=9, x=0.01, ha="left")
fig.savefig(OUT / "fig1_heldout_f1.png")
plt.close(fig)

# Figure 2: convergence of seen-client validation macro-F1
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
for ax, reg in zip(axes, REG):
    for m in FL:
        H = np.array([[r["val_macro_f1"] for r in run(reg, m, s, "history.json")] for s in SEEDS])
        x = np.arange(1, H.shape[1] + 1)
        ax.plot(x, H.mean(0), color=C[m], lw=1.6, label=N[m])
        ax.fill_between(x, H.min(0), H.max(0), color=C[m], alpha=0.13, lw=0)
    c = np.mean([run(reg, "centralized", s, "final.json")["val"]["macro_f1"] for s in SEEDS])
    ax.axhline(c, color=C["centralized"], ls="--", lw=1)
    ax.text(50, c, "centralized", ha="right", va="bottom", fontsize=7, color="#3d3d3a")
    ax.set_title(REG[reg], fontsize=8.5, loc="left")
    ax.set_xlabel("Round")
    ax.set_ylabel("Validation macro-F1 (seen clients)")
axes[0].legend(fontsize=7, ncol=2, loc="lower right")
fig.suptitle("Figure 2. Convergence (line = mean over seeds, band = min-max)", fontsize=9, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig2_convergence.png")
plt.close(fig)

# Figure 3: Amazon mechanism - predicted rating distribution (unseen) and word-weight norm
ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # one hue, ordinal (1 to 5 stars)
show = ["centralized", "fedavg", "fednova", "scaffold", "fedadam", "fedyogi"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
cm = np.array(run("amazon_vg", "centralized", 42, "final.json")["unseen"]["confusion_matrix"])
labels, shares = ["True labels"], [cm.sum(1) / cm.sum()]
for m in show:
    cm = np.mean([np.array(run("amazon_vg", m, s, "final.json")["unseen"]["confusion_matrix"]) for s in SEEDS], 0)
    labels.append(N[m])
    shares.append(cm.sum(0) / cm.sum())
shares = np.array(shares)
left = np.zeros(len(labels))
for k in range(5):
    a1.barh(range(len(labels)), shares[:, k], left=left, color=ramp[k], edgecolor="white", linewidth=1.5, label=f"{k + 1} star")
    left += shares[:, k]
a1.set_yticks(range(len(labels)))
a1.set_yticklabels(labels)
a1.invert_yaxis()
a1.set_xlim(0, 1)
a1.grid(False)
a1.set_xlabel("Share of predictions on unseen-client reviews")
a1.legend(ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
norms = [np.mean([np.linalg.norm(np.load(SCREEN / "amazon_vg" / m / f"seed{s}" / "final_weights.npz")["arr_0"]) for s in SEEDS])
         for m in show]
a2.barh(range(len(show)), norms, color=[C[m] for m in show], height=0.6)
a2.set_xscale("log")
a2.invert_yaxis()
a2.set_yticks(range(len(show)))
a2.set_yticklabels([N[m] for m in show])
a2.set_xlabel("||W|| (Frobenius norm of word weights, log scale)")
for i, v in enumerate(norms):
    a2.text(v * 1.08, i, f"{v:.1f}", va="center", fontsize=7, color="#3d3d3a")
fig.suptitle("Figure 3. Amazon: FedAvg-family models remain bias-dominated (near majority class) after 50 rounds",
             fontsize=9, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig3_amazon_mechanism.png")
plt.close(fig)

# Figure 4: Yelp Dirichlet 0.1 instability, per seed
fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
for ax, s in zip(axes, SEEDS):
    for m in ("fedavg", "scaffold", "fedadagrad"):
        v = [r["val_macro_f1"] for r in run("yelp_dir01", m, s, "history.json")]
        ax.plot(range(1, len(v) + 1), v, color=C[m], lw=1.5, label=N[m])
    ax.set_title(f"seed {s}", fontsize=8.5, loc="left")
    ax.set_xlabel("Round")
axes[0].set_ylabel("Validation macro-F1")
axes[0].legend(fontsize=7, loc="lower right")
fig.suptitle("Figure 4. Yelp Dirichlet alpha = 0.1: FedAvg oscillates; the last-round model depends on the seed",
             fontsize=9, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "fig4_dir01_instability.png")
plt.close(fig)

# Figure 5: unseen-client ECDF, Amazon, seeds pooled
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for m in ["centralized"] + FL:
    v = np.sort(np.concatenate([[u["macro_f1"] for u in run("amazon_vg", m, s, "unseen_clients.json")] for s in SEEDS]))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=C[m], lw=1.4 if m in FL else 1.2,
            ls="-" if m in FL else "--", label=N[m])
ax.set_xlabel("Per-client macro-F1, unseen product clients")
ax.set_ylabel("Fraction of clients")
ax.legend(fontsize=7, ncol=2)
ax.set_title("Figure 5. Unseen-client distribution (3 seeds pooled, 5,423 clients each)", fontsize=9, loc="left")
fig.savefig(OUT / "fig5_amazon_unseen_ecdf.png")
plt.close(fig)

# Figure 6: descriptor correlations (unseen clients); diverging, neutral grey midpoint
R = pd.read_csv(T / "descriptor_correlations.csv")
R = R[R.population == "unseen"]
desc = ["label_entropy_norm", "dominant_class_share", "mean_label", "label_tv_to_global", "n", "vocab_size",
        "mean_tokens_per_doc", "oov_token_rate", "rare_feature_mass", "centroid_cosine_dist", "centroid_js_div",
        "unique_users_share"]
meths = ["centralized", "fedavg", "fednova", "fedadam", "fedadagrad", "fedyogi"]
M = np.array([[R[(R.method == m) & (R.descriptor == k)].rho_mean.iloc[0] for m in meths] for k in desc])
cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#e8e7e2", "#c94a1d"])
fig, ax = plt.subplots(figsize=(6.2, 4.6))
im = ax.imshow(M, cmap=cmap, vmin=-0.8, vmax=0.8, aspect="auto")
ax.set_xticks(range(len(meths)))
ax.set_xticklabels([N[m] for m in meths], rotation=30, ha="right")
ax.set_yticks(range(len(desc)))
ax.set_yticklabels(desc)
ax.grid(False)
for i in range(len(desc)):
    for j in range(len(meths)):
        ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(M[i, j]) > 0.5 else "#1f1f1e")
fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman rho (mean over 3 seeds)")
ax.set_title("Figure 6. Client descriptors vs per-client macro-F1 (unseen clients)", fontsize=9, loc="left")
fig.savefig(OUT / "fig6_descriptor_correlations.png")
plt.close(fig)
print("figures written to", OUT)
