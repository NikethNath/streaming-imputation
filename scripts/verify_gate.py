"""Phase 1 gate check.

Verifies the claim "real data flowing Kafka -> window -> CSDI -> InfluxDB ->
Grafana, with the baseline imputing" -- each link independently, so a failure
says which one broke.

  python scripts/verify_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, FAIL = "PASS", "FAIL"
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((OK, name, detail))
    except Exception as e:
        results.append((FAIL, name, f"{type(e).__name__}: {e}"))


def main():
    cfg = yaml.safe_load(open("config/pipeline.yaml"))

    def dataset():
        from models.data import load_ground_truth, load_stations, geographic_adjacency
        df = load_ground_truth(cfg["data"]["ground_truth"])
        st = load_stations(cfg["data"]["latlng"])
        adj = geographic_adjacency(st)
        assert df.shape[1] == cfg["data"]["n_stations"], f"expected 36 stations, got {df.shape[1]}"
        return (f"{df.shape[0]} timesteps x {df.shape[1]} stations, "
                f"{df.isna().to_numpy().mean()*100:.1f}% natively missing, "
                f"adjacency {adj.shape} with {int((adj>0).sum())} edges")

    def kafka_up():
        from kafka import KafkaAdminClient
        a = KafkaAdminClient(bootstrap_servers=cfg["stream"]["kafka_bootstrap"])
        topics = [t for t in a.list_topics() if not t.startswith("__")]
        assert cfg["stream"]["topic_raw"] in topics, f"missing topic; have {topics}"
        return f"broker reachable, topics={sorted(topics)}"

    def kafka_has_data():
        import json
        from kafka import KafkaConsumer
        c = KafkaConsumer(cfg["stream"]["topic_raw"],
                          bootstrap_servers=cfg["stream"]["kafka_bootstrap"],
                          auto_offset_reset="earliest", consumer_timeout_ms=10000,
                          value_deserializer=lambda v: json.loads(v.decode()))
        n, masked, stations = 0, 0, set()
        for m in c:
            v = m.value
            if isinstance(v, dict) and "station" in v:
                n += 1
                stations.add(v["station"])
                masked += bool(v.get("masked"))
            if n >= 3000:
                break
        assert n > 0, "no sensor readings on the topic - run ingest/producer.py"
        return f"{n} readings, {len(stations)} stations, {masked} artificially masked"

    def influx_up():
        from influxdb_client import InfluxDBClient
        cl = InfluxDBClient(url=cfg["influx"]["url"], token=cfg["influx"]["token"],
                            org=cfg["influx"]["org"])
        assert cl.ping(), "ping failed"
        cl.close()
        return "reachable"

    def influx_has_imputations():
        from influxdb_client import InfluxDBClient
        cl = InfluxDBClient(url=cfg["influx"]["url"], token=cfg["influx"]["token"],
                            org=cfg["influx"]["org"])
        q = cl.query_api()
        b = cfg["influx"]["bucket"]
        flux = (f'from(bucket:"{b}") |> range(start:0) '
                f'|> filter(fn:(r)=>r._measurement=="pm25" and r._field=="value") '
                f'|> group(columns:["imputed"]) |> count()')
        counts = {}
        for t in q.query(flux):
            for r in t.records:
                counts[r.values.get("imputed")] = r.get_value()
        cl.close()
        assert counts.get("true", 0) > 0, f"no imputed values stored; counts={counts}"
        return f"imputed={counts.get('true',0)} observed={counts.get('false',0)}"

    def influx_has_telemetry():
        from influxdb_client import InfluxDBClient
        cl = InfluxDBClient(url=cfg["influx"]["url"], token=cfg["influx"]["token"],
                            org=cfg["influx"]["org"])
        flux = (f'from(bucket:"{cfg["influx"]["bucket"]}") |> range(start:0) '
                f'|> filter(fn:(r)=>r._measurement=="imputation_telemetry") '
                f'|> keep(columns:["_field"]) |> distinct(column:"_field")')
        fields = {r.get_value() for t in cl.query_api().query(flux) for r in t.records}
        cl.close()
        need = {"latency_ms", "nfe_per_window"}
        assert need <= fields, f"missing telemetry {need - fields}"
        return f"fields={sorted(fields)}"

    def checkpoint():
        p = Path(cfg["model"]["checkpoint"])
        assert p.exists(), f"{p} not found - run scripts/train_baseline.py"
        return f"{p} ({p.stat().st_size/1e6:.1f} MB)"

    def model_loads_and_imputes():
        import numpy as np
        from models.baseline import CSDIBaseline
        n = cfg["data"]["n_stations"]; w = cfg["data"]["window_len"]
        m = CSDIBaseline(n_steps=w, n_features=n, device="cpu", epochs=1, patience=None)
        m.load(cfg["model"]["checkpoint"])
        X = np.random.randn(1, w, n).astype(np.float32)
        X[0, 5:9, 3] = np.nan
        point, samples = m.impute(X, n_samples=2)
        assert point.shape == (1, w, n), f"bad point shape {point.shape}"
        assert samples.shape == (2, 1, w, n), f"bad sample shape {samples.shape}"
        assert not np.isnan(point).any(), "imputation left NaNs"
        return f"point{point.shape} samples{samples.shape}, gaps filled"

    def grafana():
        import urllib.request, json as j
        with urllib.request.urlopen("http://localhost:3000/api/health", timeout=10) as r:
            h = j.loads(r.read())
        assert h.get("database") == "ok", h
        return f"v{h.get('version')}, database ok"

    check("dataset: AQI-36 loads with geography", dataset)
    check("kafka: broker + topics", kafka_up)
    check("kafka: sensor readings present", kafka_has_data)
    check("influxdb: reachable", influx_up)
    check("influxdb: imputed values stored", influx_has_imputations)
    check("influxdb: telemetry stored", influx_has_telemetry)
    check("model: checkpoint exists", checkpoint)
    check("model: loads and fills gaps", model_loads_and_imputes)
    check("grafana: healthy", grafana)

    width = max(len(n) for _, n, _ in results)
    print("\n" + "=" * 78)
    print("PHASE 1 GATE CHECK".center(78))
    print("=" * 78)
    for status, name, detail in results:
        print(f"[{status}] {name.ljust(width)}  {detail}")
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    print("-" * 78)
    print(f"{len(results)-n_fail}/{len(results)} checks passed")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
