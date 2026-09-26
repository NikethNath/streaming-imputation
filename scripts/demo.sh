#!/usr/bin/env bash
# End-to-end demo: infrastructure -> stream -> baseline -> router -> dashboard.
# Designed to run inside the 10-minute limit.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

say "1/6  Infrastructure"
docker compose up -d
until [ "$(docker inspect -f '{{.State.Health.Status}}' t27-kafka 2>/dev/null)" = healthy ] \
   && [ "$(docker inspect -f '{{.State.Health.Status}}' t27-influxdb 2>/dev/null)" = healthy ]; do
  sleep 3
done
docker compose ps --format "table {{.Name}}\t{{.Status}}"

say "2/6  Dataset"
[ -s data/pm25/SampleData/pm25_ground.txt ] || ./scripts/fetch_data.sh
$PY - <<'EOF'
import sys; sys.path.insert(0,'.')
from models.data import load_ground_truth
import yaml
df = load_ground_truth(yaml.safe_load(open('config/pipeline.yaml'))['data']['ground_truth'])
print(f"AQI-36: {df.shape[0]} hourly steps x {df.shape[1]} stations, "
      f"{df.isna().to_numpy().mean()*100:.1f}% natively missing")
EOF

say "3/6  Stream a month of readings into Kafka"
$PY ingest/producer.py --speedup 1e7 --limit 120 --mode reliability_skewed

say "4/6  Baseline CSDI in the streaming path"
$PY stream/consumer.py --from-beginning --max-windows 2 --n-samples 8 \
    --stride 24 --sink influx 2>/dev/null | grep -E '^\[consumer\]'

say "5/6  Router: station-tiered equity allocation"
$PY stream/consumer.py --from-beginning --max-windows 3 --router \
    --router-mode equity --stride 24 --sink influx 2>/dev/null | grep -E '^\[consumer\]'

say "6/6  Gate check"
$PY scripts/verify_gate.py 2>/dev/null | tail -14

printf '\nDashboard: http://localhost:3000/d/t27-aqi36\n'
