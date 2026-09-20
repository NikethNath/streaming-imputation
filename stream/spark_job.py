"""Spark Structured Streaming version of the imputation consumer.

The course's Track C pipeline names Spark Structured Streaming, so this is the
path we demo. It reuses stream/windower.py and models/baseline.py rather than
reimplementing them, so the Spark and plain-Kafka paths cannot drift apart.

Design note: CSDI is a PyTorch model that wants a whole (window x station)
block, so the natural fit is foreachBatch -- Spark owns ingestion, watermarking
and micro-batching, and each micro-batch is handed to the same windowing and
imputation code the reference consumer uses. Per-station mapGroupsWithState
would fragment the window across partitions and defeat the feature attention.

Run:
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.0.1 \
      stream/spark_job.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession                                    # noqa: E402
from pyspark.sql import functions as F                                  # noqa: E402
from pyspark.sql.types import (StructType, StructField, StringType,     # noqa: E402
                               DoubleType, BooleanType)

from models.baseline import CSDIBaseline, nfe_per_window                # noqa: E402
from stream.windower import SlidingWindow                               # noqa: E402
from stream.sinks import InfluxSink, ConsoleSink                        # noqa: E402
from stream.consumer import build_scaler                                # noqa: E402
from eval import metrics                                                # noqa: E402

READING_SCHEMA = StructType([
    StructField("ts", StringType()),
    StructField("station", StringType()),
    StructField("value", DoubleType()),
    StructField("truth", DoubleType()),
    StructField("masked", BooleanType()),
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--sink", choices=["influx", "console"], default="influx")
    ap.add_argument("--starting-offsets", default="earliest")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    scaler, stations = build_scaler(cfg)
    win = cfg["data"]["window_len"]
    n_samples = args.n_samples or cfg["model"]["n_sampling_times"]

    spark = (SparkSession.builder
             .appName("team27-aqi36-imputation")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.sql.streaming.checkpointLocation", ".spark-checkpoint")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    raw = (spark.readStream
           .format("kafka")
           .option("kafka.bootstrap.servers", cfg["stream"]["kafka_bootstrap"])
           .option("subscribe", cfg["stream"]["topic_raw"])
           .option("startingOffsets", args.starting_offsets)
           .load())

    parsed = (raw.select(F.from_json(F.col("value").cast("string"),
                                     READING_SCHEMA).alias("r"))
              .select("r.*")
              .withColumn("event_time", F.to_timestamp("ts"))
              .withWatermark("event_time", "2 hours"))

    # Model and state live on the driver; foreachBatch keeps the window
    # contiguous across all 36 stations, which the feature attention needs.
    model = CSDIBaseline(n_steps=win, n_features=len(stations),
                         device=cfg["model"]["device"], epochs=1, patience=None)
    model.load(cfg["model"]["checkpoint"])
    sink = (InfluxSink(cfg["influx"]["url"], cfg["influx"]["token"],
                       cfg["influx"]["org"], cfg["influx"]["bucket"])
            if args.sink == "influx" else ConsoleSink())
    window = SlidingWindow(stations, win, stride=args.stride)
    state = {"n": 0}

    def process_batch(batch_df, batch_id):
        import time
        rows = batch_df.orderBy("event_time").collect()
        for row in rows:
            snap = window.add(row["ts"], row["station"], row["value"],
                              row["truth"], bool(row["masked"]))
            if snap is None:
                continue
            X = snap["X"][None, ...]
            t0 = time.perf_counter()
            point_s, samples_s = model.impute(scaler.transform(X), n_samples=n_samples)
            latency_ms = (time.perf_counter() - t0) * 1000

            imputed = scaler.inverse(point_s)[0]
            samples = scaler.inverse(samples_s)[:, 0]
            lo = np.quantile(samples, 0.05, axis=0)
            hi = np.quantile(samples, 0.95, axis=0)
            sink.write_window(snap["timestamps"], stations, snap["X"], imputed,
                              snap["eval_mask"], snap["truth"], lo, hi)

            em = snap["eval_mask"] & ~np.isnan(snap["truth"])
            fields = {
                "latency_ms": latency_ms,
                "nfe_per_window": float(nfe_per_window(
                    model.params["n_diffusion_steps"], n_samples)),
                "n_missing_cells": float(np.isnan(snap["X"]).sum()),
                "spark_batch_id": float(batch_id),
            }
            if em.any():
                fields["window_mae"] = metrics.mae(imputed, snap["truth"], em)
                fields["window_crps"] = metrics.crps(samples, snap["truth"], em)
            sink.write_telemetry(snap["timestamps"][-1], fields,
                                 tags={"model": "csdi-baseline", "engine": "spark"})
            state["n"] += 1
            print(f"[spark] batch={batch_id} window={state['n']} "
                  f"@ {snap['timestamps'][-1]} latency={latency_ms:.0f}ms", flush=True)

    query = (parsed.writeStream
             .foreachBatch(process_batch)
             .outputMode("append")
             .trigger(processingTime="5 seconds")
             .start())
    query.awaitTermination()


if __name__ == "__main__":
    main()
