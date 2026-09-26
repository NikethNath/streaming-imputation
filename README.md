# Streaming Imputation of Missing IoT Sensor Streams

**IOE413 — Streaming Data Analytics · Mini Project · Team 27**
Topic 5: *Diffusion Models for Real-Time Imputation of Missing IoT Sensor Streams*

An end-to-end streaming analytics pipeline that fills gaps in a live air-quality
sensor stream with a conditional diffusion model, and measures what that costs —
in latency, in compute, and in who gets a worse answer.

---

## What this is

Air-quality networks lose data constantly: nodes lose power, radios drop
packets, calibration blanks a channel. Conditional diffusion models (CSDI,
PriSTI) are the most accurate way to fill those gaps, but they were built and
measured as **offline, batch, window-at-a-time** methods. Producing one
imputation costs *thousands* of network evaluations, and every published result
is a single dataset-averaged error.

This project puts a diffusion imputer behind a real Kafka stream and measures
the two things the papers do not:

1. **What it costs per gap** — and whether we can spend less on easy gaps.
2. **Who it serves worst** — per-station error, not the average.

### Base papers

| | Paper | Venue | Role |
|---|---|---|---|
| P1 | **CSDI** — Tashiro, Song, Song, Ermon | NeurIPS 2021 | Reproduced as the baseline |
| P2 | **PriSTI** — Liu et al. | IEEE ICDE 2023 | Spatial/geographic comparison |
| P3 | **SAITS** — Du, Côté, Liu | ESWA 219 (2023) | Fast deterministic reference |

### The cost structure we are attacking

Read from the reference configs, not the papers:

| | Diffusion steps | Posterior samples | **Evaluations per window** |
|---|---|---|---|
| CSDI | 50 | 100 | **5,000** |
| PriSTI | 100 | 100 | **10,000** |

The model is small — 414,081 trainable parameters. The cost is that the network
runs five to ten thousand times per window. So there are **two** things to
economise on, not one: chain length *and* sample count. Both are exposed in
`models/baseline.py`.

---

## Architecture

```
  AQI-36 archive (36 Beijing PM2.5 stations, hourly, 12 months)
        |
        |  ingest/producer.py  — replays in timestamp order at controllable
        |                        speed, injects point/block/skewed missingness
        v
  [ Kafka topic: aqi36.raw ]                          <- apache/kafka 3.9 (KRaft)
        |
        |  stream/spark_job.py    (Spark Structured Streaming, Track C path)
        |  stream/consumer.py     (plain Kafka consumer, reference path)
        |      both share stream/windower.py + models/baseline.py
        v
  sliding window (36 timesteps x 36 stations)
        |
        |  models/baseline.py — CSDI via PyPOTS
        v
  imputed values + 90% uncertainty band + per-window telemetry
        |
        v
  [ InfluxDB bucket: aqi36 ]                          <- influxdb 2.7
        |
        v
  [ Grafana dashboard ]                               <- grafana 11.3
     observed vs imputed · latency · evaluations/window · MAE/CRPS
     · per-station fairness panel
```

**Storage note.** The approved proposal specified TimescaleDB. We use
**InfluxDB** instead: it is the store named in the course's own Track C
pipeline, and it provisions into Grafana with less glue. The storage engine is
an implementation detail, not part of the approved scientific scope.

---

## Quick start

Requires Docker, Docker Compose, and Python 3.11+.

```bash
# 1. Infrastructure
docker compose up -d
#    Kafka    -> localhost:9092
#    InfluxDB -> localhost:8086   (team27 / team27password, org ioe413)
#    Grafana  -> localhost:3000   (admin / admin)

# 2. Python environment (CPU-only torch; this machine has no CUDA GPU)
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -r requirements.txt

# 3. Train the baseline
.venv/bin/python scripts/train_baseline.py --smoke     # ~1 min, proves the path
./scripts/train_overnight.sh                            # full quality, several hours

# 4. Run the pipeline (two terminals)
.venv/bin/python ingest/producer.py --speedup 3600
.venv/bin/python stream/consumer.py --from-beginning --n-samples 30

# 5. Open the dashboard
xdg-open http://localhost:3000/d/t27-aqi36

# 6. Verify every link in the chain
.venv/bin/python scripts/verify_gate.py
```

### Spark path

```bash
.venv/bin/pip install pyspark
python stream/spark_job.py                 # self-resolving connector
# or:
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0 \
  stream/spark_job.py
```

The job declares the Kafka connector itself, so plain `python stream/spark_job.py`
works too; the JAR resolves from Maven on first run.

> **Java version.** Verified working on **Java 25** with Spark 4.2.0, which is
> newer than Spark documents support for. If a different JDK gives trouble,
> Java 21 is the safe choice. `stream/consumer.py` is the engine-independent
> path and produces identical output and telemetry.

---

## Repository layout

```
config/pipeline.yaml     central configuration (data, stream, influx, model)
data/pm25/               AQI-36 archive + station coordinates
ingest/
  missingness.py         point / block / sensor-failure / reliability-skewed processes
  producer.py            archive -> Kafka, in timestamp order
stream/
  windower.py            sliding window state shared by both engines
  sinks.py               InfluxDB and console sinks
  consumer.py            reference Kafka consumer
  spark_job.py           Spark Structured Streaming job
models/
  data.py                loading, month splits, windowing, geographic adjacency
  baseline.py            CSDI via PyPOTS  [INTERFACE FROZEN]
eval/
  metrics.py             MAE / RMSE / MRE / CRPS / PICP, per-station breakdown
  equity.py              worst-decile, Gini, error-vs-missingness, AQI bands
scripts/
  fetch_data.sh          download the AQI-36 benchmark
  train_baseline.py      train + score + write results/baseline_metrics.json
  train_overnight.sh     full-quality training run
  verify_gate.py         end-to-end Phase 1 gate check
dashboards/              Grafana provisioning and dashboard JSON
```

---

## Dataset

**AQI-36** — hourly PM2.5 from 36 Beijing stations over 12 months (8,760
timesteps), the archive released with ST-MVL (Yi et al., IJCAI 2016). This is
the dataset **both CSDI and PriSTI** evaluate on, which is what makes our
numbers comparable to their published tables without re-deriving anything.

We keep their month split — test on months 3, 6, 9, 12 — and their 36-step
window. Training uses a stride-1 sliding view (2,917 windows); validation and
test use non-overlapping windows so each evaluated cell is scored once.

The archive already contains ~13% genuinely missing values. `ingest/missingness.py`
layers further gaps on top so we have ground truth to score against:

| Protocol | What it models |
|---|---|
| `point` | MCAR dropout |
| `block` | bursty per-station outage (radio/power) |
| `mixed` | both, the default |
| `reliability_skewed` | **ours** — heavy-tailed per-station failure rates, so a minority of stations carry most of the missingness |

That last one is the realistic case, and the one where a dataset-average metric
hides the damage.

---

## Metrics

**Accuracy** — MAE, RMSE, MRE, CRPS (quantile-loss estimator, as in CSDI/PriSTI).
Scored **only on artificially masked cells**, where we know the truth.

**Calibration** — PICP at 90% and mean interval width. A fast sampler that is no
longer calibrated has given up the only advantage diffusion had over SAITS, and
MAE will not reveal it.

**Cost** — per-window latency, and **network function evaluations per window**
(`steps × samples`), the hardware-independent unit.

**Equity** — per-station MAE distribution, worst-decile MAE, max–min gap, Gini
coefficient of error, and the regression slope of error against each station's
missing rate. A positive slope is the precise statement *"the model is worst
where data is scarcest."* Plus **AQI-band misclassification**: how often an
imputed value lands in the wrong public-health category, which is the error a
health department would actually act on.

---

## Phase 1 results

Baseline CSDI, held-out months (3, 6, 9, 12) of AQI-36, 9,557 artificially
masked cells scored. This is a deliberately **simplified** reproduction — 16
epochs, stride-4 training view, 20 posterior samples — which the brief permits.

| Metric | Ours | CSDI published (AQI-36) |
|---|---|---|
| MAE | 14.43 µg/m³ | 9.51 |
| RMSE | 31.18 | — |
| MRE | 0.197 | — |
| CRPS | 0.150 | 0.1056 |
| PICP @ 90% | 0.794 | — |

Same order of magnitude as the published result, which is what a simplified
reproduction should achieve. Undercoverage at 90% is expected from a short run.

### The equity finding

The audit that none of the three base papers performs, on the same run:

| | |
|---|---|
| Best station MAE | 8.23 |
| Worst station MAE | 34.44 |
| **Worst / best** | **4.2×** |
| Worst-decile MAE | 27.35 |
| Gini of per-station error | 0.176 |
| **Error vs missing-rate slope** | **+70.8 (r = +0.59)** |
| AQI-band misclassification | **27.3%** |

The slope is the headline: **a station's imputation error correlates strongly
with how much of its data is missing** (r = +0.59). The model is measurably
worst exactly where data is scarcest — and a dataset-average MAE of 14.43 makes
that invisible. Over a quarter of imputed values land in the wrong public-health
band.

### Cost

At CSDI's own defaults the baseline needs **565 seconds per window** on CPU at
1,000 evaluations per window. That number is the entire motivation for the
Phase 2 router.

## Phase 2 results: budget allocation

### A free 2.5x saving (positive)

Sweeping diffusion steps against posterior samples, **the cost-accuracy frontier
never uses 50 steps**:

| steps x samples | NFE | MAE | CRPS |
|---|---|---|---|
| 20 x 16 | **320** | **12.95** | 0.1867 |
| 50 x 16 | 800 | 13.09 | 0.1891 |

Better accuracy for 2.5x less compute, with no adaptivity at all. At matched
cost the sample axis also buys more than the step axis: 20x8 (160 NFE, MAE
13.36) beats 50x4 (200 NFE, MAE 13.74).

### Per-gap triage does not work (negative, and we show why)

An oracle allocator appears to beat the best uniform budget by **49%**. That
headroom is an artefact of the sampler's own randomness.

| quantity | test-retest correlation across two runs |
|---|---|
| per-gap **error** | **+0.76 … +0.91** — difficulty is real structure |
| per-gap **gain** | **+0.015** — what a router must predict, and it is noise |

Allocating on one run and scoring on an **independent** run turns the oracle's
+32.5% into **-7.2%**, i.e. worse than not adapting. Learned routers confirm it:
held-out R2 on gain is negative at every budget.

**If you are building an adaptive-compute router over a stochastic model, run
this two-run check before you trust an oracle bound.** It costs two sampling
passes: correlate per-item error across runs, correlate per-item *gain* across
runs, and score the oracle's allocation on the held-out run.

### Selective imputation: knowing what not to publish (positive)

The failed router leaves behind a useful asymmetry. Per-gap **gain** is
unpredictable, but per-gap **error** is (held-out R2 = +0.214). So instead of
spending compute where it helps, we decline to publish what we cannot stand
behind. For an alerting service a missing value is more honest than a
confidently wrong one.

Withholding the worst-predicted 20% cuts MAE among the published 80% from 12.22
to **9.70** (-20.6%), capturing 51% of what an oracle could achieve.

But ranking by error magnitude is the **wrong criterion for decisions**. A band
flip is caused by proximity to a threshold, not by error size: a 2 ug/m3 error
at the 35.4 boundary changes the public-health category, a 20 ug/m3 error
mid-band does not. So we rank instead by **posterior band risk** -- the
fraction of posterior samples landing in a different AQI band than the
published median:

| coverage | by predicted error | **by posterior band risk** | oracle |
|---|---|---|---|
| 100% | 33.0% | 33.0% | 33.0% |
| 90% | 32.3% | 29.6% | 25.6% |
| **80%** | 31.8% | **27.1%** | 16.3% |
| 50% | 31.2% | **20.0%** | 0.0% |

An 18% relative reduction in wrong public-health categories at 80% coverage,
against essentially nothing from the error-based ranking. It costs no extra
compute -- the samples are already drawn.

**This is the one result that requires a generative imputer.** SAITS returns a
point estimate and cannot express band risk at all. It is also the clearest
answer to what diffusion buys over a cheaper deterministic model.

Live in the pipeline via `--abstain-risk 0.4`, which withholds ~200 high-risk
cells per window and records `n_withheld` and `mean_band_risk` as telemetry.

### Equity allocation works (positive)

Station-level error *is* reliable, so the lever moves up a level — the
worst-served 30% of stations get the expensive budget, the rest the cheap one:

| variant | NFE | MAE | worst-10% | **Gini** |
|---|---|---|---|---|
| uniform cheap (10x4) | 40 | 15.30 | 24.00 | 0.166 |
| uniform rich (20x16) | 320 | 13.46 | 21.30 | 0.156 |
| **equity mix** | **126** | 14.76 | 22.44 | **0.155** |

The equity mix achieves the **best error Gini of the three at 2.5x less compute
than uniform-rich**, capturing ~58% of the worst-decile improvement for 31% of
the added cost.

Live in the pipeline: **217 NFE/window vs the baseline's 400**, with stations
promoted online as the deficit tracker observes error.

## Reproducing the results

```bash
./scripts/demo.sh                       # full end-to-end demo, <10 min

python scripts/budget_sweep.py          # the (steps, samples) frontier
python scripts/build_delta_dataset.py   # per-gap errors at each budget
python router/train_router.py           # oracle bound + learned router
python scripts/gain_reliability.py      # the two-run diagnostic
python scripts/equity_experiment.py     # station-level allocation
python scripts/selective_imputation.py  # risk-coverage for error
python scripts/selective_bands.py       # decision-aware abstention
python scripts/make_figures.py          # report figures
```

## Report

`report/main.tex` builds a 6-page IEEE two-column paper
(`pdflatex main.tex` twice). Both the PDF and the editable source are tracked.

## Status

- [x] Ingestion, storage and dashboard layer
- [x] Kafka -> window -> CSDI -> InfluxDB -> Grafana loop closed
- [x] CSDI baseline trained and scored against the published result
- [x] Metrics and equity audit
- [x] Spark Structured Streaming path verified against Kafka
- [x] Budget-controllable sampler, validated against the reference
- [x] Per-gap router + oracle bound + reliability diagnostic
- [x] Station-level equity allocation, live in the serving path
- [x] Common-random-numbers test of the negative result (does not rescue it)
- [x] Selective imputation with posterior band risk, live in the serving path
- [x] IEEE report and figures
- [x] `scripts/verify_gate.py` — 9/9 checks passing

## Tool versions

| Tool | Version |
|---|---|
| Python | 3.14 |
| PyTorch | 2.14.0 (CPU) |
| PyPOTS | 1.5 |
| Apache Kafka | 3.9.0 (KRaft) |
| Apache Spark | 4.2.0 |
| InfluxDB | 2.7 |
| Grafana | 11.3.0 |

## References

1. Tashiro, Song, Song & Ermon. *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation.* NeurIPS 2021. arXiv:2107.03502
2. Liu, Huang, Feng, Sun, Du & Fu. *PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation.* IEEE ICDE 2023. arXiv:2302.09746
3. Du, Côté & Liu. *SAITS: Self-Attention-based Imputation for Time Series.* ESWA 219 (2023) 119619. arXiv:2202.08516
4. Yi, Zheng, Zhang & Li. *ST-MVL: Filling Missing Values in Geo-sensory Time Series Data.* IJCAI 2016. — source of the AQI-36 benchmark
