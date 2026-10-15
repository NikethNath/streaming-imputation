"""Sweep the (diffusion steps, posterior samples) grid on held-out AQI-36.

This produces the accuracy-versus-cost frontier reachable by turning the two
*global* knobs -- the thing CSDI's own future-work section proposes (ODE
solvers) and the thing the survey discusses (DDIM / DPM-Solver). Every point
here applies one budget uniformly to every window.

The router has to beat this frontier, not just beat the full-budget default.
Anything less is not a contribution.

Writes results/budget_sweep.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.data import (load_ground_truth, split_by_month, to_windows,     # noqa: E402
                         Standardiser)
from models.baseline import CSDIBaseline                                     # noqa: E402
from models import sampler                                                   # noqa: E402
from eval import metrics, equity                                             # noqa: E402


def load_eval_windows(cfg, n_windows, mask_rate, seed, split="test"):
    df = load_ground_truth(cfg["data"]["ground_truth"])
    stations = [str(c) for c in df.columns]
    win = cfg["data"]["window_len"]
    train_df, val_df, test_df = split_by_month(df)
    scaler = Standardiser().fit(to_windows(train_df, win, stride=1))
    src = {"test": test_df, "val": val_df}[split]
    W = scaler.transform(to_windows(src, win, stride=win))
    if n_windows:
        W = W[:n_windows]
    rng = np.random.default_rng(seed)
    observed = ~np.isnan(W)
    hide = (rng.random(W.shape) < mask_rate) & observed
    corrupt = W.copy()
    corrupt[hide] = np.nan
    return scaler, stations, W, corrupt, hide, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--n-windows", type=int, default=12)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--steps", type=int, nargs="+", default=[2, 5, 10, 20, 50])
    ap.add_argument("--samples", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--out", default="results/budget_sweep.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    scaler, stations, W, corrupt, hide, df = load_eval_windows(
        cfg, args.n_windows, args.mask_rate, args.seed, args.split)
    truth = scaler.inverse(W)
    miss_rate = {s: float(np.isnan(df[s]).mean()) for s in stations}
    print(f"{args.split}: {W.shape[0]} windows, {int(hide.sum())} masked cells")

    rows = []
    model = CSDIBaseline(n_steps=cfg["data"]["window_len"], n_features=len(stations),
                         device=cfg["model"]["device"], epochs=1, patience=None)
    model.load(cfg["model"]["checkpoint"])

    for S in args.steps:
        for N in args.samples:
            t0 = time.perf_counter()
            smp, nfe = sampler.sample(model, corrupt, n_steps=S, n_samples=N, mode="ddim")
            secs = time.perf_counter() - t0
            pt = scaler.inverse(sampler.point_estimate(smp))
            sm = scaler.inverse(smp)
            row = {
                "steps": S, "samples": N, "nfe": nfe,
                "MAE": metrics.mae(pt, truth, hide),
                "RMSE": metrics.rmse(pt, truth, hide),
                "CRPS": metrics.crps(sm, truth, hide) if N > 1 else None,
                "PICP90": metrics.picp(sm, truth, hide) if N > 1 else None,
                "seconds_total": secs,
                "seconds_per_window": secs / W.shape[0],
            }
            ps = metrics.per_station(pt, truth, hide, stations)
            row["equity"] = equity.audit(ps, miss_rate)
            rows.append(row)
            crps_txt = f"{row['CRPS']:.4f}" if row["CRPS"] is not None else "  --  "
            print(f"  steps={S:2d} samples={N:2d} nfe={nfe:4d}  "
                  f"MAE={row['MAE']:6.2f}  CRPS={crps_txt}  "
                  f"{row['seconds_per_window']:.2f}s/win", flush=True)

    Path("results").mkdir(exist_ok=True)
    payload = {"config": vars(args), "rows": rows,
               "n_masked_cells": int(hide.sum()), "n_windows": int(W.shape[0])}
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")

    # Pareto frontier on (nfe, MAE)
    best = sorted(rows, key=lambda r: (r["nfe"], r["MAE"]))
    frontier, best_mae = [], float("inf")
    for r in best:
        if r["MAE"] < best_mae:
            best_mae = r["MAE"]
            frontier.append(r)
    print("\n=== global-knob Pareto frontier (nfe -> best MAE) ===")
    for r in frontier:
        print(f"  nfe={r['nfe']:5d}  steps={r['steps']:2d} samples={r['samples']:2d}  "
              f"MAE={r['MAE']:.2f}")


if __name__ == "__main__":
    main()
