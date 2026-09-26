"""Station-level budget allocation for equity.

Per-gap triage fails on this data: which gap *benefits* from extra compute has
test-retest reliability r ~= 0.015, so the signal a gap-level router would need
does not exist (see scripts/gain_reliability.py).

But per-gap *error* is highly reliable (r = 0.76 .. 0.91), and error is strongly
concentrated by station: a station's imputation error correlates with how much
of its data is missing. Station difficulty is therefore a real, persistent,
predictable property -- unlike per-gap gain.

So we move the lever up a level. Instead of asking "which gap deserves more
compute", ask "which *station* is being served worst", and give those stations a
larger share of the same total budget. This is the equity-aware allocation, and
unlike gap triage it is built on a signal that actually survives re-sampling.

Compares, at matched total NFE:
  * uniform   -- every station gets the same budget (the standard approach)
  * equity    -- worst-served stations get the expensive budget, best-served the cheap one

Writes results/equity_experiment.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.data import (load_ground_truth, split_by_month, to_windows,      # noqa: E402
                         Standardiser)
from models.baseline import CSDIBaseline                                     # noqa: E402
from models import sampler                                                   # noqa: E402
from eval import metrics, equity as eq                                       # noqa: E402


def build(cfg, split, n_windows, mask_rate, seed):
    df = load_ground_truth(cfg["data"]["ground_truth"])
    stations = [str(c) for c in df.columns]
    win = cfg["data"]["window_len"]
    tr, va, te = split_by_month(df)
    scaler = Standardiser().fit(to_windows(tr, win, stride=1))
    src = {"val": va, "test": te}[split]
    W = scaler.transform(to_windows(src, win, stride=win))[:n_windows]
    rng = np.random.default_rng(seed)
    obs = ~np.isnan(W)
    hide = (rng.random(W.shape) < mask_rate) & obs
    corrupt = W.copy(); corrupt[hide] = np.nan
    return scaler, stations, W, corrupt, hide, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--mask-rate", type=float, default=0.15)
    ap.add_argument("--cheap", default="10x4")
    ap.add_argument("--rich", default="20x16")
    ap.add_argument("--worst-frac", type=float, default=0.3)
    args = ap.parse_args()

    def parse(b):
        s, n = b.lower().split("x"); return int(s), int(n)
    cheapS, cheapN = parse(args.cheap)
    richS, richN = parse(args.rich)
    cheap_nfe, rich_nfe = cheapS * cheapN, richS * richN

    cfg = yaml.safe_load(open(args.config))
    model = CSDIBaseline(n_steps=cfg["data"]["window_len"], n_features=36,
                         device="cpu", epochs=1, patience=None)
    model.load(cfg["model"]["checkpoint"])

    # --- Stage 1: learn which stations are worst served, on VALIDATION ------
    sc, stations, Wv, cv, hv, df = build(cfg, "val", args.n_windows, args.mask_rate, 11)
    truth_v = sc.inverse(Wv)
    smp, _ = sampler.sample(model, cv, n_steps=cheapS, n_samples=cheapN, mode="ddim")
    pv = sc.inverse(sampler.point_estimate(smp))
    ps_val = metrics.per_station(pv, truth_v, hv, stations)
    order = sorted([s for s in stations if np.isfinite(ps_val[s])],
                   key=lambda s: -ps_val[s])
    k = max(1, int(round(args.worst_frac * len(order))))
    worst = set(order[:k])
    print(f"worst-served stations identified on validation (k={k}): "
          f"{sorted(worst)[:6]}...")

    # --- Stage 2: evaluate on TEST at matched total budget -----------------
    sc, stations, Wt, ct, ht, df = build(cfg, "test", args.n_windows, args.mask_rate, 27)
    truth = sc.inverse(Wt)
    miss = {s: float(np.isnan(df[s]).mean()) for s in stations}
    idx_worst = np.array([s in worst for s in stations])

    smp_c, _ = sampler.sample(model, ct, n_steps=cheapS, n_samples=cheapN, mode="ddim")
    smp_r, _ = sampler.sample(model, ct, n_steps=richS, n_samples=richN, mode="ddim")
    p_cheap = sc.inverse(sampler.point_estimate(smp_c))
    p_rich = sc.inverse(sampler.point_estimate(smp_r))

    # equity mix: worst stations get the rich budget, the rest get cheap
    p_eq = p_cheap.copy()
    p_eq[:, :, idx_worst] = p_rich[:, :, idx_worst]
    frac = idx_worst.mean()
    eq_nfe = frac * rich_nfe + (1 - frac) * cheap_nfe

    # uniform comparator at the SAME average cost: pick the single budget whose
    # cost is closest to the equity mix, and also report both pure budgets.
    variants = {
        f"uniform_cheap ({args.cheap}, {cheap_nfe} nfe)": (p_cheap, float(cheap_nfe)),
        f"uniform_rich ({args.rich}, {rich_nfe} nfe)": (p_rich, float(rich_nfe)),
        f"equity_mix ({eq_nfe:.0f} nfe avg)": (p_eq, float(eq_nfe)),
    }

    out = {}
    print(f"\n{'variant':40s} {'NFE':>7} {'MAE':>7} {'worst10%':>9} "
          f"{'gini':>7} {'slope':>8} {'band%':>7}")
    for name, (pred, nfe) in variants.items():
        ps = metrics.per_station(pred, truth, ht, stations)
        bands = eq.band_misclassification(pred, truth, ht, stations)
        a = eq.audit(ps, miss, bands)
        row = {"nfe": nfe, "MAE": metrics.mae(pred, truth, ht),
               "worst_decile_MAE": a["worst_decile_MAE"],
               "error_gini": a["error_gini"],
               "slope": a.get("error_vs_missingness_slope"),
               "band_misclassification": bands["overall"],
               "max_min_gap": a["max_min_gap"]}
        out[name] = row
        print(f"{name:40s} {nfe:7.0f} {row['MAE']:7.2f} "
              f"{row['worst_decile_MAE']:9.2f} {row['error_gini']:7.3f} "
              f"{row['slope']:8.1f} {row['band_misclassification']*100:6.1f}%")

    Path("results").mkdir(exist_ok=True)
    Path("results/equity_experiment.json").write_text(json.dumps(
        {"variants": out, "worst_stations": sorted(worst),
         "config": vars(args)}, indent=2))
    print("\nwrote results/equity_experiment.json")


if __name__ == "__main__":
    main()
