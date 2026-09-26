"""Streaming imputation consumer: Kafka -> sliding window -> CSDI -> InfluxDB.

This is the reference consumer that closes the Phase 1 loop. The Spark
Structured Streaming job in stream/spark_job.py wraps the same windowing and
imputation code behind Spark's stateful API; both share stream/windower.py and
models/baseline.py so the two paths cannot drift apart.

Per-window telemetry (latency, NFE, error on artificially masked cells) is
written alongside the values, because those are the numbers the Phase 2 router
is judged on.
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
from models.data import (load_ground_truth, Standardiser, split_by_month,     # noqa: E402
                         to_windows, load_stations, geographic_adjacency)
from models.baseline import CSDIBaseline, nfe_per_window                              # noqa: E402
from stream.windower import SlidingWindow                                             # noqa: E402
from stream.sinks import InfluxSink, ConsoleSink                                       # noqa: E402
from eval import metrics                                                               # noqa: E402


def build_scaler(cfg):
    """Refit the training-split scaler so the consumer matches training exactly."""
    df = load_ground_truth(cfg["data"]["ground_truth"])
    train_df, _, _ = split_by_month(df)
    return Standardiser().fit(to_windows(train_df, cfg["data"]["window_len"], stride=1)), \
        [str(c) for c in df.columns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="posterior samples per window; overrides config")
    ap.add_argument("--stride", type=int, default=6,
                    help="emit an imputed window every N timesteps")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--sink", choices=["influx", "console"], default="influx")
    ap.add_argument("--from-beginning", action="store_true")
    ap.add_argument("--router", action="store_true",
                    help="enable budget allocation in the serving path")
    ap.add_argument("--router-mode", default="equity", choices=["equity", "gain"],
                    help="equity = station-tiered (shipped); gain = per-gap "
                         "triage, kept to reproduce the negative result")
    ap.add_argument("--router-model", default="results/router.pkl",
                    help="gain-mode model bundle")
    ap.add_argument("--nfe-budget", type=float, default=2000,
                    help="network function evaluations allowed per window")
    ap.add_argument("--no-equity", action="store_true",
                    help="disable the equity reservation (ablation)")
    ap.add_argument("--abstain-risk", type=float, default=None,
                    help="withhold imputed cells whose posterior band risk "
                         "exceeds this (0-1); e.g. 0.4. Publishing a value the "
                         "posterior disagrees about can place a station in the "
                         "wrong public-health band.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    scaler, stations = build_scaler(cfg)
    win = cfg["data"]["window_len"]
    n_samples = args.n_samples or cfg["model"]["n_sampling_times"]

    print(f"[consumer] loading checkpoint {cfg['model']['checkpoint']}")
    # patience must be < epochs in PyPOTS; we only load a checkpoint here so
    # neither matters, but both must be internally consistent.
    model = CSDIBaseline(n_steps=win, n_features=len(stations),
                         device=cfg["model"]["device"], epochs=1, patience=None)
    model.load(cfg["model"]["checkpoint"])

    router = None
    if args.router:
        from router.runtime import GapRouter
        adj = geographic_adjacency(load_stations(cfg["data"]["latlng"]))
        model_path = args.router_model if args.router_mode == "gain" else None
        router = GapRouter(model_path, adj, mode=args.router_mode)
        print(f"[consumer] router enabled mode={args.router_mode} "
              f"equity={'off' if args.no_equity else 'on'}")

    if args.sink == "influx":
        sink = InfluxSink(cfg["influx"]["url"], cfg["influx"]["token"],
                          cfg["influx"]["org"], cfg["influx"]["bucket"])
    else:
        sink = ConsoleSink()

    from kafka import KafkaConsumer
    consumer = KafkaConsumer(
        cfg["stream"]["topic_raw"],
        bootstrap_servers=cfg["stream"]["kafka_bootstrap"],
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="earliest" if args.from_beginning else "latest",
        group_id="t27-imputer",
        consumer_timeout_ms=120_000,
    )

    window = SlidingWindow(stations, win, stride=args.stride)
    n_done = 0
    budget_txt = (f"router={args.router_mode}" if router is not None
                  else f"n_samples={n_samples}")
    print(f"[consumer] listening on {cfg['stream']['topic_raw']} "
          f"(window={win}, stride={args.stride}, {budget_txt})")

    for msg in consumer:
        m = msg.value
        if not isinstance(m, dict) or "station" not in m or "ts" not in m:
            continue  # ignore anything that is not a sensor reading
        snap = window.add(m["ts"], m["station"], m.get("value"),
                          m.get("truth"), m.get("masked", False))
        if snap is None:
            continue

        X = snap["X"][None, ...]                       # (1, win, n_stations)
        Xs = scaler.transform(X)
        t0 = time.perf_counter()
        if router is not None:
            imputed_s, info = router.impute_window(
                model, Xs[0], args.nfe_budget, use_equity=not args.no_equity)
            latency_ms = (time.perf_counter() - t0) * 1000
            imputed = scaler.inverse(imputed_s)
            # Uncertainty band still needs samples; draw a small set at the
            # cheapest routed budget so the dashboard stays informative.
            from models import sampler as _s
            smp, band_nfe = _s.sample(model, Xs, n_steps=10, n_samples=8, mode="ddim")
            samples = scaler.inverse(smp)[:, 0]
            nfe_used = info["nfe_used"] + band_nfe
        else:
            point_s, samples_s = model.impute(Xs, n_samples=n_samples)
            latency_ms = (time.perf_counter() - t0) * 1000
            imputed = scaler.inverse(point_s)[0]
            samples = scaler.inverse(samples_s)[:, 0]
            info = {}
            nfe_used = nfe_per_window(model.params["n_diffusion_steps"], n_samples)

        lo = np.quantile(samples, 0.05, axis=0)
        hi = np.quantile(samples, 0.95, axis=0)

        # Decision risk: fraction of the posterior landing in a different AQI
        # band than the value we would publish. Cheap -- the samples exist.
        from eval.equity import band_of
        band_risk = (band_of(samples) != band_of(imputed)[None, ...]).mean(axis=0)
        n_withheld = 0
        if args.abstain_risk is not None:
            was_missing = np.isnan(snap["X"])
            drop = was_missing & (band_risk > args.abstain_risk)
            n_withheld = int(drop.sum())
            imputed = np.where(drop, np.nan, imputed)

        sink.write_window(snap["timestamps"], stations, snap["X"], imputed,
                          snap["eval_mask"], snap["truth"], lo, hi)

        em = snap["eval_mask"] & ~np.isnan(snap["truth"]) & ~np.isnan(imputed)
        fields = {
            "latency_ms": latency_ms,
            "nfe_per_window": float(nfe_used),
            "n_missing_cells": float(np.isnan(snap["X"]).sum()),
            "n_scoreable_cells": float(em.sum()),
            "mean_band_risk": float(band_risk[np.isnan(snap["X"])].mean())
                              if np.isnan(snap["X"]).any() else 0.0,
            "n_withheld": float(n_withheld),
        }
        for k in ("n_gaps", "escalation_rate", "n_priority_gaps",
                  "n_promoted_stations"):
            if k in info:
                fields[k] = float(info[k])
        if em.any():
            fields["window_mae"] = metrics.mae(imputed, snap["truth"], em)
            fields["window_crps"] = metrics.crps(samples, snap["truth"], em)
            if router is not None:
                # feed per-station error back so the equity reservation adapts
                ps = metrics.per_station(imputed, snap["truth"], em,
                                         list(range(len(stations))))
                router.observe_errors(ps)
        sink.write_telemetry(snap["timestamps"][-1], fields,
                             tags={"model": "csdi-router" if router else "csdi-baseline"})

        n_done += 1
        mae_txt = f"{fields.get('window_mae', float('nan')):.2f}"
        extra = (f" nfe={nfe_used} withheld={n_withheld}"
                 f" promoted={info.get('n_promoted_stations', '-')}"
                 f" escal={info.get('escalation_rate',0)*100:.0f}%"
                 if router is not None else f" nfe={nfe_used} withheld={n_withheld}")
        print(f"[consumer] window {n_done} @ {snap['timestamps'][-1]} "
              f"latency={latency_ms:.0f}ms mae={mae_txt} "
              f"missing={int(fields['n_missing_cells'])}{extra}", flush=True)

        if args.max_windows and n_done >= args.max_windows:
            break

    sink.close()
    print(f"[consumer] done, {n_done} windows imputed")


if __name__ == "__main__":
    main()
