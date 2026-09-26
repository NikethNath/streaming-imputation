"""Is the oracle's headroom real, or is it sampling noise?

The oracle allocator knows each gap's realised error at every budget and beats
the uniform knob by ~49%. A learned router gets nowhere near that, and the gain
models have negative held-out R2. Before concluding "the features are bad", we
have to ask whether the quantity being predicted is predictable *at all*.

Diffusion sampling is stochastic. Run the same gap at the same budget twice and
you get two different errors. Any per-gap allocator -- learned or oracle -- that
is fitted to one realisation is partly fitting that run's noise.

This script measures it directly:

  1. test-retest reliability: correlate per-gap error across two independent
     runs at the same budget. This is the ceiling on how well *anything* can
     predict per-gap error.
  2. honest oracle: allocate using run A's errors, then score that allocation
     on run B. The difference between this and the self-scored oracle is the
     amount of the "headroom" that was noise.

Writes results/gain_reliability.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.data import (load_ground_truth, load_stations, split_by_month,   # noqa: E402
                         to_windows, Standardiser, geographic_adjacency)
from models.baseline import CSDIBaseline                                     # noqa: E402
from models import sampler                                                   # noqa: E402
from router import features as feat                                          # noqa: E402
from eval.oracle import allocate_lagrangian, uniform_choice                  # noqa: E402


def gap_errors(preds, truth, hide, corrupt, adj, budgets):
    """Per-gap mean absolute error for each budget."""
    rows, meta = [], []
    for w in range(corrupt.shape[0]):
        _, keys = feat.features_for_window(corrupt[w], adj)
        for (station, start, length) in keys:
            seg = slice(start, start + length)
            m = hide[w, seg, station]
            if not m.any():
                continue
            t = truth[w, seg, station][m]
            rows.append([float(np.abs(preds[b][w, seg, station][m] - t).mean())
                         for b in budgets])
            meta.append((w, station, start, length))
    return np.array(rows), np.array(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--n-windows", type=int, default=24)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--budgets", default="5x1,10x4,20x8,20x16")
    args = ap.parse_args()

    budgets = []
    for b in args.budgets.split(","):
        s, n = b.lower().split("x")
        budgets.append((int(s), int(n)))
    budgets.sort(key=lambda b: b[0] * b[1])
    costs = np.array([s * n for s, n in budgets], dtype=float)

    cfg = yaml.safe_load(open(args.config))
    df = load_ground_truth(cfg["data"]["ground_truth"])
    stations = [str(c) for c in df.columns]
    win = cfg["data"]["window_len"]
    adj = geographic_adjacency(load_stations(cfg["data"]["latlng"]))
    train_df, val_df, _ = split_by_month(df)
    scaler = Standardiser().fit(to_windows(train_df, win, stride=1))
    W = scaler.transform(to_windows(val_df, win, stride=win))[:args.n_windows]

    rng = np.random.default_rng(27)
    observed = ~np.isnan(W)
    hide = (rng.random(W.shape) < args.mask_rate) & observed
    corrupt = W.copy(); corrupt[hide] = np.nan
    truth = scaler.inverse(W)

    model = CSDIBaseline(n_steps=win, n_features=len(stations), device="cpu",
                         epochs=1, patience=None)
    model.load(cfg["model"]["checkpoint"])

    runs = []
    for run in (0, 1):
        np.random.seed(1000 + run)
        import torch; torch.manual_seed(1000 + run)
        preds = {}
        for i, (S, N) in enumerate(budgets):
            smp, _ = sampler.sample(model, corrupt, n_steps=S, n_samples=N, mode="ddim")
            preds[i] = scaler.inverse(sampler.point_estimate(smp))
            print(f"  run {run} budget {S}x{N} done", flush=True)
        E, meta = gap_errors(preds, truth, hide, corrupt, adj, range(len(budgets)))
        runs.append(E)

    A, Bm = runs
    n = A.shape[0]
    print(f"\ngaps: {n}")

    out = {"budgets": [f"{s}x{n_}" for s, n_ in budgets], "n_gaps": int(n),
           "reliability": {}, "oracle": {}}

    print("\n=== test-retest reliability of per-gap error (same budget, 2 runs) ===")
    print("  a value near 0 means per-gap error is mostly sampling noise")
    for i, (S, N) in enumerate(budgets):
        r = float(np.corrcoef(A[:, i], Bm[:, i])[0, 1])
        out["reliability"][f"{S}x{N}"] = r
        print(f"  budget {S:2d}x{N:2d}  r = {r:+.3f}")

    print("\n=== reliability of the GAIN itself (what the router must predict) ===")
    gA = A[:, 0:1] - A
    gB = Bm[:, 0:1] - Bm
    for i in range(1, len(budgets)):
        r = float(np.corrcoef(gA[:, i], gB[:, i])[0, 1])
        out["reliability"][f"gain_{budgets[i][0]}x{budgets[i][1]}"] = r
        print(f"  gain at {budgets[i][0]:2d}x{budgets[i][1]:2d}  r = {r:+.3f}")

    print("\n=== self-scored oracle vs honest oracle ===")
    print("  self-scored: allocate on run A, score on run A (what we reported)")
    print("  honest:      allocate on run A, score on run B (no noise reuse)")
    rows = []
    for frac in (0.1, 0.25, 0.5, 0.75):
        budget = costs.min() * n + frac * (costs.max() - costs.min()) * n
        idx, cost, _ = allocate_lagrangian(A, costs, budget)
        self_mae = float(A[np.arange(n), idx].mean())
        honest_mae = float(Bm[np.arange(n), idx].mean())
        ub, _, usum = uniform_choice(Bm, costs, budget)
        uni_mae = float(usum / n)
        rows.append({"budget_per_gap": float(budget / n), "self_oracle": self_mae,
                     "honest_oracle": honest_mae, "uniform": uni_mae,
                     "self_gain_pct": 100 * (uni_mae - self_mae) / uni_mae,
                     "honest_gain_pct": 100 * (uni_mae - honest_mae) / uni_mae})
        print(f"  nfe/gap={budget/n:7.1f}  uniform={uni_mae:7.3f}  "
              f"self-oracle={self_mae:7.3f} ({rows[-1]['self_gain_pct']:+5.1f}%)  "
              f"honest-oracle={honest_mae:7.3f} ({rows[-1]['honest_gain_pct']:+5.1f}%)")
    out["oracle"] = rows

    Path("results").mkdir(exist_ok=True)
    Path("results/gain_reliability.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/gain_reliability.json")


if __name__ == "__main__":
    main()
