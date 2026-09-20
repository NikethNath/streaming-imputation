"""Replay AQI-36 into Kafka as a live per-station sensor stream.

The course brief accepts "a Kafka/MQTT topic replaying a public dataset" as a
real stream; a static CSV read once is explicitly not acceptable. So this walks
the archive in timestamp order at a controllable speed and emits one message per
station per timestep, applying an artificial missingness process on top of the
gaps already in the data.

The `truth` field exists only so the consumer can score itself online. A real
deployment would not have it, and nothing in the imputation path reads it.
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
from ingest.missingness import MissingnessSimulator          # noqa: E402
from models.data import load_ground_truth                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--speedup", type=float, default=None,
                    help="data-seconds per real second; overrides config")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N timesteps (for smoke tests)")
    ap.add_argument("--mode", default="mixed",
                    choices=["point", "block", "mixed", "reliability_skewed"])
    ap.add_argument("--dry-run", action="store_true",
                    help="print to stdout instead of publishing to Kafka")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    df = load_ground_truth(cfg["data"]["ground_truth"])
    stations = [str(c) for c in df.columns]
    speedup = args.speedup or cfg["stream"]["speedup"]

    mcfg = cfg["stream"]["missing"]
    sim = MissingnessSimulator(
        n_stations=len(stations),
        seed=mcfg["seed"],
        point_rate=mcfg["point_rate"],
        block_rate=mcfg["block_rate"],
        block_len_mean=mcfg["block_len_mean"],
        mode=args.mode,
    )

    producer = None
    if not args.dry_run:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=cfg["stream"]["kafka_bootstrap"],
            value_serializer=lambda v: json.dumps(v).encode(),
            key_serializer=lambda k: k.encode(),
            linger_ms=20,
        )

    topic = cfg["stream"]["topic_raw"]
    # AQI-36 is hourly, so one timestep is 3600 data-seconds.
    sleep_per_step = 3600.0 / speedup
    sent = 0
    t0 = time.time()

    for i, (ts, row) in enumerate(df.iterrows()):
        if args.limit and i >= args.limit:
            break
        hide = sim.step()
        values = row.to_numpy(dtype=float)
        for j, sid in enumerate(stations):
            truth = values[j]
            observed_in_archive = not np.isnan(truth)
            masked = bool(hide[j]) and observed_in_archive
            value = None if (masked or not observed_in_archive) else float(truth)
            msg = {
                "ts": ts.isoformat(),
                "station": sid,
                "value": value,
                "truth": float(truth) if observed_in_archive else None,
                "masked": masked,           # artificially hidden -> scoreable
            }
            if args.dry_run:
                if j == 0:
                    print(json.dumps(msg))
            else:
                producer.send(topic, key=sid, value=msg)
            sent += 1

        if i % 24 == 0:
            rate = sent / max(time.time() - t0, 1e-9)
            print(f"[producer] t={ts} step={i} msgs={sent} ({rate:.0f}/s)",
                  file=sys.stderr, flush=True)
        if sleep_per_step > 0:
            time.sleep(sleep_per_step)

    if producer:
        producer.flush()
    print(f"[producer] done, {sent} messages", file=sys.stderr)


if __name__ == "__main__":
    main()
