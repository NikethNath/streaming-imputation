#!/usr/bin/env bash
# Overnight: train the full-quality baseline, then re-run every experiment on it.
#
# The current results came from a 16-epoch, stride-4 model. This trains for up
# to 60 epochs on the stride-1 view (CSDI's own setting), then repeats the
# experiments so every number in the report comes from the better model. The
# old results are kept in results/v1_short/ for comparison.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14
PY=.venv/bin/python
CKPT=checkpoints/csdi_aqi36_full.pypots
CFG=config/pipeline_full.yaml
step() { printf '\n[%s] === %s ===\n' "$(date '+%F %T')" "$1"; }

step "backing up current results to results/v1_short"
mkdir -p results/v1_short
cp results/*.json results/*.npz results/*.pkl results/*.log results/v1_short/ 2>/dev/null || true

step "training (up to 60 epochs, stride 1, early stopping patience 10)"
if ! $PY scripts/train_baseline.py --epochs 60 --batch-size 16 --train-stride 1 \
     --val-windows 16 --eval-steps 50 --eval-samples 50 \
     --checkpoint-out "$CKPT" --metrics-out results/baseline_metrics_full.json; then
  step "TRAINING FAILED - stopping"; exit 1
fi
cp results/baseline_metrics_full.json results/baseline_metrics.json

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
