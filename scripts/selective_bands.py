"""Does abstention stop us publishing the wrong public-health category?

MAE improvements are abstract. The number an air-quality service acts on is the
AQI band: an imputed PM2.5 value that lands in the wrong band can suppress an
alert that should have fired. This script measures band misclassification among
the values we actually publish, as a function of how many we withhold.

Uses the error model trained on validation gaps, applied to held-out test gaps.
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
from eval import equity as eq                                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--data", default="results/delta_dataset.npz")
    ap.add_argument("--n-windows", type=int, default=20)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    # --- train the error model on the validation-built gap dataset ---------
    d = np.load(args.data, allow_pickle=True)
    Xtr, Etr = d["X"], d["errors"]
    names = [str(n) for n in d["feature_names"]]
    from lightgbm import LGBMRegressor
    em = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                       min_child_samples=20, random_state=27, verbose=-1)
    em.fit(Xtr, Etr[:, 4])
    print(f"error model trained on {len(Xtr)} validation gaps")

    # --- apply to held-out TEST windows -----------------------------------
    cfg = yaml.safe_load(open(args.config))
    df = load_ground_truth(cfg["data"]["ground_truth"])
    stations = [str(c) for c in df.columns]
    win = cfg["data"]["window_len"]
    adj = geographic_adjacency(load_stations(cfg["data"]["latlng"]))
    tr_df, _, te_df = split_by_month(df)
    scaler = Standardiser().fit(to_windows(tr_df, win, stride=1))
    W = scaler.transform(to_windows(te_df, win, stride=win))[:args.n_windows]
    rng = np.random.default_rng(27)
    obs = ~np.isnan(W)
    hide = (rng.random(W.shape) < args.mask_rate) & obs
    corrupt = W.copy(); corrupt[hide] = np.nan
    truth = scaler.inverse(W)

    model = CSDIBaseline(n_steps=win, n_features=len(stations), device=cfg["model"]["device"],
                         epochs=1, patience=None)
    model.load(cfg["model"]["checkpoint"])
    smp, nfe = sampler.sample(model, corrupt, n_steps=args.steps,
                              n_samples=args.samples, mode="ddim")
    pred = scaler.inverse(sampler.point_estimate(smp))
    print(f"imputed {W.shape[0]} test windows at {args.steps}x{args.samples} "
          f"({nfe} NFE)")

    # Posterior band risk: what fraction of the posterior samples land in a
    # different AQI band than the published median? This is a direct,
    # calibration-free estimate of decision risk, and it is exactly what the
    # diffusion posterior provides and a deterministic imputer cannot.
    samples_raw = scaler.inverse(smp)            # (N, W, L, K)
    band_samples = eq.band_of(samples_raw)
    band_median = eq.band_of(pred)
    band_risk_full = (band_samples != band_median[None, ...]).mean(axis=0)

    # gather per-cell records grouped by gap, with that gap's predicted error
    vals_p, vals_t, gap_pred, cell_risk = [], [], [], []
    for w in range(W.shape[0]):
        feats, keys = feat.features_for_window(corrupt[w], adj)
        if not feats:
            continue
        pe = em.predict(feat.to_matrix(feats, names))
        for gi, (st, start, length) in enumerate(keys):
            seg = slice(start, start + length)
            m = hide[w, seg, st]
            if not m.any():
                continue
            p = pred[w, seg, st][m]; t = truth[w, seg, st][m]
            vals_p.append(p); vals_t.append(t)
            gap_pred.append(np.full(p.shape, pe[gi]))
            cell_risk.append(band_risk_full[w, seg, st][m])
    P = np.concatenate(vals_p); T = np.concatenate(vals_t)
    G = np.concatenate(gap_pred)
    Rk = np.concatenate(cell_risk)
    print(f"evaluated cells: {len(P)}")

    band_wrong = (eq.band_of(P) != eq.band_of(T))
    abs_err = np.abs(P - T)
    base_band = float(band_wrong.mean()); base_mae = float(abs_err.mean())
    strategies = {
        "predicted_error": np.argsort(G),      # rank by learned error magnitude
        "posterior_band_risk": np.argsort(Rk), # rank by posterior band disagreement
        "oracle_band": np.argsort(band_wrong.astype(float)),
    }

    rows = []
    print(f"\n{'coverage':>9} {'by MAE-model':>13} {'by band-risk':>13} "
          f"{'oracle':>9} {'MAE(band-risk)':>15}")
    for cov in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = max(1, int(round(cov * len(P))))
        r = {"coverage": cov}
        for name, order in strategies.items():
            keep = order[:k]
            r[f"band_{name}"] = float(band_wrong[keep].mean())
            r[f"mae_{name}"] = float(abs_err[keep].mean())
        r["band_reduction_pct"] = 100 * (base_band - r["band_posterior_band_risk"]) / base_band
        rows.append(r)
        print(f"{cov*100:8.0f}% {r['band_predicted_error']*100:12.1f}% "
              f"{r['band_posterior_band_risk']*100:12.1f}% "
              f"{r['band_oracle_band']*100:8.1f}% "
              f"{r['mae_posterior_band_risk']:15.2f}")

    Path("results").mkdir(exist_ok=True)
    Path("results/selective_bands.json").write_text(json.dumps(
        {"budget": f"{args.steps}x{args.samples}", "n_cells": int(len(P)),
         "base_band_misclassification": base_band, "base_MAE": base_mae,
         "rows": rows}, indent=2))
    print("\nwrote results/selective_bands.json")


if __name__ == "__main__":
    main()
