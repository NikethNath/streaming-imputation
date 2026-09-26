"""Build the (gap features -> value of computation) dataset.

For each gap we record the error achieved at every budget on the grid. The
router's target is

    delta = error(cheapest budget) - error(this budget)

i.e. how much this gap actually gains from being given more compute. Gaps with
delta near zero should be routed cheap; gaps with large delta should be
escalated. Learning this is the whole enhancement.

Built on the *validation* months so the router never sees the test split.
Writes results/delta_dataset.npz.
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
from models.data import (load_ground_truth, load_stations, split_by_month,   # noqa: E402
                         to_windows, Standardiser, geographic_adjacency)
from models.baseline import CSDIBaseline                                     # noqa: E402
from models import sampler                                                   # noqa: E402
from router import features as feat                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--n-windows", type=int, default=24)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--budgets", default="2x1,5x2,10x4,20x8,50x16",
                    help="comma separated steps x samples")
    ap.add_argument("--out", default="results/delta_dataset.npz")
    ap.add_argument("--crn-seed", type=int, default=None,
                    help="fix the initial latent across budgets (common random numbers)")
    args = ap.parse_args()

    budgets = []
    for b in args.budgets.split(","):
        s, n = b.lower().split("x")
        budgets.append((int(s), int(n)))
    budgets.sort(key=lambda b: b[0] * b[1])
    print("budgets (cheap -> expensive):",
          [f"{s}x{n}={s*n}nfe" for s, n in budgets])

    cfg = yaml.safe_load(open(args.config))
    df = load_ground_truth(cfg["data"]["ground_truth"])
    stations = [str(c) for c in df.columns]
    win = cfg["data"]["window_len"]
    adj = geographic_adjacency(load_stations(cfg["data"]["latlng"]))

    train_df, val_df, _ = split_by_month(df)
    scaler = Standardiser().fit(to_windows(train_df, win, stride=1))
    W = scaler.transform(to_windows(val_df, win, stride=win))[:args.n_windows]

    rng = np.random.default_rng(args.seed)
    observed = ~np.isnan(W)
    hide = (rng.random(W.shape) < args.mask_rate) & observed
    corrupt = W.copy()
    corrupt[hide] = np.nan
    truth = scaler.inverse(W)
    print(f"val windows={W.shape[0]}  masked cells={int(hide.sum())}")

    model = CSDIBaseline(n_steps=win, n_features=len(stations), device="cpu",
                         epochs=1, patience=None)
    model.load(cfg["model"]["checkpoint"])

    # error per gap, per budget
    preds = {}
    for (S, N) in budgets:
        t0 = time.perf_counter()
        smp, _ = sampler.sample(model, corrupt, n_steps=S, n_samples=N, mode="ddim",
                                crn_seed=args.crn_seed)
        preds[(S, N)] = scaler.inverse(sampler.point_estimate(smp))
        print(f"  budget {S}x{N} done in {time.perf_counter()-t0:.0f}s", flush=True)

    rows_X, rows_err, rows_meta = [], [], []
    for w in range(W.shape[0]):
        feats, keys = feat.features_for_window(corrupt[w], adj)
        if not feats:
            continue
        Xmat = feat.to_matrix(feats)
        for gi, (station, start, length) in enumerate(keys):
            seg = slice(start, start + length)
            m = hide[w, seg, station]
            if not m.any():
                continue          # no ground truth in this gap -> unscoreable
            t = truth[w, seg, station][m]
            errs = []
            for b in budgets:
                p = preds[b][w, seg, station][m]
                errs.append(float(np.abs(p - t).mean()))
            rows_X.append(Xmat[gi])
            rows_err.append(errs)
            rows_meta.append((w, station, start, length, int(m.sum())))

    X = np.array(rows_X, dtype=np.float32)
    E = np.array(rows_err, dtype=np.float32)
    meta = np.array(rows_meta, dtype=np.int32)
    delta = E[:, 0:1] - E                    # gain relative to cheapest budget

    Path("results").mkdir(exist_ok=True)
    np.savez(args.out, X=X, errors=E, delta=delta, meta=meta,
             budgets=np.array(budgets), feature_names=np.array(feat.FEATURE_NAMES))
    print(f"\ngaps with ground truth: {len(X)}")
    print(f"feature matrix: {X.shape}")
    print("\nmean |error| by budget:")
    for i, (S, N) in enumerate(budgets):
        print(f"  {S:2d}x{N:2d} ({S*N:4d} nfe)  MAE={E[:, i].mean():.3f}  "
              f"mean gain vs cheapest={delta[:, i].mean():+.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
