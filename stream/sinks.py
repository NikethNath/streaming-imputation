"""InfluxDB sink for raw readings, imputed values and per-window telemetry."""
from __future__ import annotations

import numpy as np


class InfluxSink:
    def __init__(self, url: str, token: str, org: str, bucket: str):
        from influxdb_client import InfluxDBClient
        from influxdb_client.client.write_api import SYNCHRONOUS
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.bucket = bucket
        self.org = org

    def write_window(self, timestamps, stations, observed, imputed,
                     eval_mask, truth, lo=None, hi=None):
        """Persist one imputed window.

        Raw and imputed series both land in InfluxDB so the Grafana panel can
        overlay them, and so the gate check can confirm both are stored.
        """
        from influxdb_client import Point
        points = []
        for i, ts in enumerate(timestamps):
            for j, sid in enumerate(stations):
                was_missing = bool(np.isnan(observed[i, j]))
                # A NaN imputation is a value the band-risk check held back;
                # InfluxDB cannot store NaN, so record it as its own series.
                withheld = was_missing and bool(np.isnan(imputed[i, j]))
                kind = "withheld" if withheld else ("true" if was_missing else "false")
                p = Point("pm25").tag("station", sid).tag("imputed", kind).time(ts)
                p = p.field("withheld", 1.0) if withheld else p.field("value", float(imputed[i, j]))
                if not was_missing:
                    p = p.field("observed", float(observed[i, j]))
                if bool(eval_mask[i, j]) and not np.isnan(truth[i, j]) and not withheld:
                    p = p.field("truth", float(truth[i, j]))
                    p = p.field("abs_error", abs(float(imputed[i, j]) - float(truth[i, j])))
                if lo is not None:
                    p = p.field("lo90", float(lo[i, j])).field("hi90", float(hi[i, j]))
                points.append(p)
        self.write_api.write(bucket=self.bucket, org=self.org, record=points)

    def write_telemetry(self, ts, fields: dict, tags: dict = None):
        from influxdb_client import Point
        p = Point("imputation_telemetry")
        for k, v in (tags or {}).items():
            p = p.tag(k, str(v))
        for k, v in fields.items():
            if v is not None and np.isfinite(v):
                p = p.field(k, float(v))
        self.write_api.write(bucket=self.bucket, org=self.org, record=p.time(ts))

    def close(self):
        self.client.close()


class ConsoleSink:
    """Fallback sink so the pipeline is testable without InfluxDB running."""

    def write_window(self, timestamps, stations, observed, imputed,
                     eval_mask, truth, lo=None, hi=None):
        n = int(np.isnan(observed).sum())
        print(f"[sink] window {timestamps[0]}..{timestamps[-1]} imputed {n} cells")

    def write_telemetry(self, ts, fields, tags=None):
        print(f"[sink] telemetry {ts} {fields}")

    def close(self):
        pass
