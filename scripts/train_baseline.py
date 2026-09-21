"""Train the CSDI baseline on AQI-36 and score it on the held-out months.

Usage:
  python scripts/train_baseline.py --epochs 100
  python scripts/train_baseline.py --smoke        # 2 epochs, tiny, for CI
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
                         to_windows, Standardiser)
from models.baseline import CSDIBaseline, nfe_per_window                      # noqa: E402
from eval import metrics, equity                                              # noqa: E402


def mask_for_eval(windows: np.ndarray, rate: float, seed: int = 27):
    """Hide a fraction of observed cells so we have ground truth to score on."""
    rng = np.random.default_rng(seed)
    observed = ~np.isnan(windows)
    hide = (rng.random(windows.shape) < rate) & observed
    corrupted = windows.copy()
    corrupted[hide] = np.nan
    return corrupted, hide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-samples", type=int, default=50)
    ap.add_argument("--mask-rate", type=float, default=0.10)
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--val-windows", type=int, default=16,
                    help="cap the validation set; CSDI validation runs the "
                         "full diffusion sampler, so a large val set "
                         "dominates epoch time")
    ap.add_argument("--smoke", action="store_true", help="tiny run to prove the path works")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    df = load_ground_truth(cfg["data"]["ground_truth"])
    stations = [str(c) for c in df.columns]
    win = cfg["data"]["window_len"]

    train_df, val_df, test_df = split_by_month(df)
    print(f"rows  train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # stride 1 on train for data volume; non-overlapping on val/test so each
    # evaluated cell is scored exactly once.
    train_w = to_windows(train_df, win, stride=args.train_stride)
    val_w = to_windows(val_df, win, stride=win)
    test_w = to_windows(test_df, win, stride=win)
    print(f"windows train={train_w.shape} val={val_w.shape} test={test_w.shape}")

    # Validation runs the diffusion sampler, so it costs far more per window
    # than training does. Cap it -- we only need enough windows to rank
    # checkpoints, not to produce a publishable number. Leaving this
    # uncapped took 96 minutes per epoch.
    if args.val_windows and len(val_w) > args.val_windows:
        val_w = val_w[:args.val_windows]
        print(f'validation capped to {len(val_w)} windows')

    scaler = Standardiser().fit(train_w)
    train_s = scaler.transform(train_w)
    val_s = scaler.transform(val_w)
    test_s = scaler.transform(test_w)

    if args.smoke:
        args.epochs = 2
        args.eval_samples = 5
        train_s = train_s[:32]
        val_s = val_s[:8]
        test_s = test_s[:8]
        print("SMOKE MODE: 2 epochs, 32 train windows, 5 posterior samples")

    # Corrupt the test set so we have scoreable ground truth.
    test_corrupt, eval_mask = mask_for_eval(test_s, args.mask_rate)

    model = CSDIBaseline(
        n_steps=win,
        n_features=len(stations),
        device=cfg["model"]["device"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=None if args.smoke else 10,
    )

    t0 = time.time()
    model.train(train_s, val_s)
    train_secs = time.time() - t0
    print(f"trained in {train_secs:.1f}s")

    t0 = time.time()
    point_s, samples_s = model.impute(test_corrupt, n_samples=args.eval_samples)
    infer_secs = time.time() - t0

    # Back to ug/m3 before scoring, so MAE is comparable to published tables.
    point = scaler.inverse(point_s)
    samples = scaler.inverse(samples_s)
    truth = scaler.inverse(test_s)

    res = metrics.summarise(point, truth, eval_mask, samples, station_ids=stations)
    per_station = res.pop("per_station_MAE")

    miss_rate = {sid: float(np.isnan(df[sid]).mean()) for sid in stations}
    bands = equity.band_misclassification(point, truth, eval_mask, stations)
    audit = equity.audit(per_station, miss_rate, bands)

    n_windows = test_corrupt.shape[0]
    res["train_seconds"] = round(train_secs, 1)
    res["inference_seconds"] = round(infer_secs, 2)
    res["seconds_per_window"] = round(infer_secs / max(n_windows, 1), 3)
    res["nfe_per_window"] = nfe_per_window(
        model.params["n_diffusion_steps"], args.eval_samples)

    print("\n=== baseline metrics (ug/m3) ===")
    for k, v in res.items():
        print(f"  {k:24s} {v}")
    print("\n=== equity audit ===")
    for k, v in audit.items():
        print(f"  {k:30s} {v:.4f}" if isinstance(v, float) else f"  {k:30s} {v}")

    out = Path("results"); out.mkdir(exist_ok=True)
    payload = {"metrics": res, "equity": audit, "per_station_MAE": per_station,
               "missing_rate": miss_rate, "config": vars(args)}
    (out / "baseline_metrics.json").write_text(json.dumps(payload, indent=2))
    ckpt = cfg["model"]["checkpoint"]
    model.save(ckpt)
    print(f"\nsaved checkpoint -> {ckpt}")
    print(f"saved metrics    -> results/baseline_metrics.json")


if __name__ == "__main__":
    main()
