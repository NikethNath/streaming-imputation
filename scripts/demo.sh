#!/usr/bin/env bash
# End-to-end demo for the video: infrastructure -> Kafka -> Spark + CSDI ->
# router -> band-risk abstention -> InfluxDB -> Grafana. About 3 minutes of
# compute; it pauses between steps so the presenter can talk.
#
#   ./scripts/demo.sh             # pauses for Enter between steps
#   ./scripts/demo.sh --no-pause  # runs straight through
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PAUSE=1; [ "${1:-}" = "--no-pause" ] && PAUSE=0
DASH="http://localhost:3000/d/t27-aqi36?orgId=1&var-station=${STATION:-001010}"

say()   { printf '\n\033[1;7m %s \033[0m\n' "$1"; }
note()  { printf '\033[2m%s\033[0m\n' "$1"; }
pause() { [ "$PAUSE" = 1 ] && { printf '\n\033[2m[Enter to continue]\033[0m'; read -r _; } || true; }

say "STEP 0  Clean start"
docker compose up -d >/dev/null 2>&1
until [ "$(docker inspect -f '{{.State.Health.Status}}' t27-kafka 2>/dev/null)" = healthy ] \
   && [ "$(docker inspect -f '{{.State.Health.Status}}' t27-influxdb 2>/dev/null)" = healthy ]; do sleep 2; done
docker exec t27-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 \
  --delete --topic aqi36.raw >/dev/null 2>&1 || true
docker exec t27-influxdb influx delete --org ioe413 --bucket aqi36 --token team27-dev-token \
  --start 1970-01-01T00:00:00Z --stop 2100-01-01T00:00:00Z >/dev/null
rm -rf .spark-checkpoint
note "Kafka topic and InfluxDB bucket emptied."

say "STEP 1  Infrastructure: Kafka, InfluxDB, Grafana (Docker Compose)"
docker compose ps --format "table {{.Name}}\t{{.Image}}\t{{.Status}}"
pause

say "STEP 2  Dataset: AQI-36, Beijing PM2.5"
$PY - <<'EOF'
import sys; sys.path.insert(0, '.')
import warnings; warnings.filterwarnings('ignore')
import yaml
from models.data import load_ground_truth
df = load_ground_truth(yaml.safe_load(open('config/pipeline.yaml'))['data']['ground_truth'])
print(f"{df.shape[1]} stations x {df.shape[0]} hourly readings, "
      f"{df.isna().to_numpy().mean()*100:.1f}% already missing")
print(f"first hours, first 6 stations:\n{df.iloc[:5, :6].round(0).to_string()}")
EOF
pause

say "STEP 3  Producer: replay 5 days of sensor readings into Kafka, injecting outages"
$PY ingest/producer.py --speedup 1e7 --limit 120 --mode reliability_skewed 2>&1 | grep -E "^\[producer\]" | tail -3
note "Three messages as they sit in the Kafka topic (masked=true is an injected outage):"
docker exec t27-kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 \
  --topic aqi36.raw --from-beginning --max-messages 3 2>/dev/null
pause

say "STEP 4  Spark Structured Streaming + CSDI: impute 36-hour windows"
note "Spark reads the topic, builds windows, and runs CSDI in foreachBatch (50 steps x 8 samples = 400 NFE)."
$PY stream/spark_job.py --max-windows 3 --n-samples 8 --stride 24 --sink influx 2>/dev/null \
  | grep --line-buffered -E '^\[spark\]'
pause

say "STEP 5  Router: station-tiered compute budget"
note "Worst-served stations get a bigger budget; everyone else gets the cheap one."
$PY stream/consumer.py --from-beginning --max-windows 3 --router --router-mode equity \
  --stride 24 --sink influx 2>/dev/null | grep --line-buffered -E '^\[consumer\] (router|window)'
pause

say "STEP 6  Main novelty: band-risk abstention"
note "Values whose posterior samples disagree about the AQI band (risk > 0.3) are held back, not published."
$PY stream/consumer.py --from-beginning --max-windows 3 --n-samples 16 --abstain-risk 0.3 \
  --stride 24 --sink influx 2>/dev/null | grep --line-buffered -E '^\[consumer\] window'
pause

say "STEP 7  Check: raw and imputed series are both stored in InfluxDB"
$PY scripts/verify_gate.py 2>/dev/null | tail -12

printf '\nDashboard: %s\n' "$DASH"
command -v xdg-open >/dev/null && xdg-open "$DASH" >/dev/null 2>&1 || true
