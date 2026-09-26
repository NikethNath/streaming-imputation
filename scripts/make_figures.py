"""Generate the report figures from the saved experiment results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("report/figures"); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight"})

# ---------------------------------------------------------------- Fig 1
sweep = json.load(open("results/budget_sweep.json"))
rows = [r for r in sweep["rows"] if r["MAE"] < 100]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for S in sorted({r["steps"] for r in rows}):
    rs = sorted([r for r in rows if r["steps"] == S], key=lambda r: r["nfe"])
    ax.plot([r["nfe"] for r in rs], [r["MAE"] for r in rs],
            marker="o", ms=3, lw=1.1, label=f"{S} steps")
ax.set_xscale("log"); ax.set_xlabel("NFE per window (log)")
ax.set_ylabel(r"MAE ($\mu$g/m$^3$)")
ax.set_title("Cost-accuracy frontier", fontsize=8)
ax.legend(fontsize=6, ncol=2)
fig.savefig(OUT / "pareto.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 2
rel = json.load(open("results/gain_reliability.json"))
err_r = {k: v for k, v in rel["reliability"].items() if not k.startswith("gain_")}
gain_r = {k.replace("gain_", ""): v for k, v in rel["reliability"].items()
          if k.startswith("gain_")}
labels = [k for k in err_r if k in gain_r]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
x = np.arange(len(labels)); w = 0.36
ax.bar(x - w/2, [err_r[k] for k in labels], w, label="per-gap error")
ax.bar(x + w/2, [gain_r[k] for k in labels], w, label="per-gap gain")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("test-retest correlation $r$")
ax.set_xlabel("budget (steps $\\times$ samples)")
ax.set_title("What is reproducible across runs", fontsize=8)
ax.legend(fontsize=6)
fig.savefig(OUT / "reliability.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 3
oc = rel["oracle"]
fig, ax = plt.subplots(figsize=(3.4, 2.3))
b = [r["budget_per_gap"] for r in oc]
ax.plot(b, [r["uniform"] for r in oc], marker="s", ms=3, lw=1.1, label="uniform budget")
ax.plot(b, [r["self_oracle"] for r in oc], marker="o", ms=3, lw=1.1,
        ls="--", label="oracle (self-scored)")
ax.plot(b, [r["honest_oracle"] for r in oc], marker="^", ms=3, lw=1.1,
        label="oracle (honest, held-out run)")
ax.set_xlabel("NFE per gap"); ax.set_ylabel(r"MAE ($\mu$g/m$^3$)")
ax.set_title("The oracle's headroom is noise", fontsize=8)
ax.legend(fontsize=6)
fig.savefig(OUT / "oracle.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 4
eqx = json.load(open("results/equity_experiment.json"))["variants"]
names = list(eqx)
short = ["uniform\ncheap", "uniform\nrich", "equity\nmix"]
fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.2))
axes[0].bar(short, [eqx[n]["worst_decile_MAE"] for n in names],
            color=["#888", "#555", "#1f77b4"])
axes[0].set_ylabel(r"worst-decile MAE ($\mu$g/m$^3$)")
axes[0].tick_params(labelsize=6)
axes[1].bar(short, [eqx[n]["nfe"] for n in names],
            color=["#888", "#555", "#1f77b4"])
axes[1].set_ylabel("NFE per window"); axes[1].tick_params(labelsize=6)
fig.suptitle("Equity allocation: fairness per unit compute", fontsize=8)
fig.savefig(OUT / "equity.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 5
bm = json.load(open("results/baseline_metrics.json"))
ps = bm["per_station_MAE"]; mr = bm["missing_rate"]
ks = [k for k in ps if np.isfinite(ps[k])]
x = np.array([mr[k] * 100 for k in ks]); y = np.array([ps[k] for k in ks])
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.scatter(x, y, s=12, alpha=0.75)
m, c = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(xs, m * xs + c, "r-", lw=1.2,
        label=f"slope={m:.2f}, r={np.corrcoef(x,y)[0,1]:+.2f}")
ax.set_xlabel("station missing rate (%)")
ax.set_ylabel(r"station MAE ($\mu$g/m$^3$)")
ax.set_title("Error concentrates where data is scarce", fontsize=8)
ax.legend(fontsize=6)
fig.savefig(OUT / "equity_scatter.pdf"); plt.close(fig)

# ---------------------------------------------------------------- Fig 6
sb = json.load(open("results/selective_bands.json"))
si = json.load(open("results/selective_imputation.json"))
rows = sb["rows"]
cov = [r["coverage"] * 100 for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
axes[0].plot(cov, [r["band_predicted_error"] * 100 for r in rows],
             marker="s", ms=3, lw=1.1, label="rank by predicted error")
axes[0].plot(cov, [r["band_posterior_band_risk"] * 100 for r in rows],
             marker="o", ms=3, lw=1.3, label="rank by posterior band risk")
axes[0].plot(cov, [r["band_oracle_band"] * 100 for r in rows],
             marker="^", ms=3, lw=1.0, ls="--", color="grey", label="oracle")
axes[0].set_xlabel("coverage: % of gaps published")
axes[0].set_ylabel("AQI band misclassification (%)")
axes[0].set_title("Decision-aware abstention", fontsize=8)
axes[0].legend(fontsize=6); axes[0].invert_xaxis()

srows = si["rows"]
scov = [r["coverage"] * 100 for r in srows]
axes[1].plot(scov, [r["model_MAE"] for r in srows], marker="o", ms=3, lw=1.3,
             label="learned error model")
axes[1].plot(scov, [r["oracle_MAE"] for r in srows], marker="^", ms=3, lw=1.0,
             ls="--", color="grey", label="oracle")
axes[1].axhline(si["base_MAE"], color="k", lw=0.8, ls=":", label="impute everything")
axes[1].set_xlabel("coverage: % of gaps published")
axes[1].set_ylabel(r"MAE among published ($\mu$g/m$^3$)")
axes[1].set_title("Risk-coverage for error", fontsize=8)
axes[1].legend(fontsize=6); axes[1].invert_xaxis()
fig.savefig(OUT / "selective.pdf"); plt.close(fig)

print("figures written to", OUT)
for f in sorted(OUT.glob("*.pdf")):
    print("  ", f.name, f.stat().st_size, "bytes")
