#!/usr/bin/env bash
# Overnight v3: retrain CSDI the way the paper trains it on AQI-36, then re-run
# every experiment on the new model.
#
# Two changes from run_overnight.sh:
#   * target_strategy=mix -- half the training masks are random cells, half are
#     copied from the dataset's own outage patterns, so the model also learns
#     long sensor outages, not only isolated missing hours;
#   * the learning rate drops 10x at epochs 25 and 40, as in CSDI's recipe.
# The previous results are kept in results/v2_full/ so the two models can be
# compared before either is used in the report.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14
PY=.venv/bin/python
CKPT=checkpoints/csdi_aqi36_mix.pypots
CFG=config/pipeline_mix.yaml
step() { printf '\n[%s] === %s ===\n' "$(date '+%F %T')" "$1"; }

step "backing up current results to results/v2_full"
mkdir -p results/v2_full
cp results/*.json results/*.npz results/*.pkl results/*.log results/v2_full/ 2>/dev/null || true
cp -r report/figures results/v2_full/figures 2>/dev/null || true

step "training (mix masks, LR decay at 25/40, up to 60 epochs, patience 15)"
if ! $PY scripts/train_baseline.py --epochs 60 --batch-size 16 --train-stride 1 \
     --val-windows 16 --eval-steps 50 --eval-samples 50 \
     --target-strategy mix --lr-decay-epochs 25,40 --patience 15 \
     --checkpoint-out "$CKPT" --metrics-out results/baseline_metrics_mix.json; then
  step "TRAINING FAILED - stopping"; exit 1
fi
cp results/baseline_metrics_mix.json results/baseline_metrics.json

sed "s|checkpoint: .*|checkpoint: $CKPT|" config/pipeline.yaml > "$CFG"

run() { step "$1"; shift; "$@" || echo "!!! step failed, continuing"; }
run "budget sweep"            $PY scripts/budget_sweep.py --config "$CFG" --n-windows 12
run "delta dataset"           $PY scripts/build_delta_dataset.py --config "$CFG" --n-windows 30 --budgets 5x1,5x4,10x4,20x4,20x8,20x16
run "delta dataset (CRN)"     $PY scripts/build_delta_dataset.py --config "$CFG" --n-windows 30 --budgets 5x1,5x4,10x4,20x4,20x8,20x16 --crn-seed 4242 --out results/delta_dataset_crn.npz
run "router + oracle"         $PY router/train_router.py
run "router (CRN)"            $PY router/train_router.py --data results/delta_dataset_crn.npz --out results/router_crn.pkl
run "gain reliability"        $PY scripts/gain_reliability.py --config "$CFG"
run "equity allocation"       $PY scripts/equity_experiment.py --config "$CFG"
run "selective imputation"    $PY scripts/selective_imputation.py
run "selective bands"         $PY scripts/selective_bands.py --config "$CFG"
run "figures"                 $PY scripts/make_figures.py
step "ALL DONE"
