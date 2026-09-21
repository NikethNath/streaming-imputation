#!/usr/bin/env bash
# Full-quality baseline training. Several hours on CPU -- start it before you
# stop for the night. The gate only needs the fast run; this is the one whose
# numbers go in the report.
#
# Note --val-windows: CSDI validation runs the full diffusion sampler, so an
# uncapped validation set costs ~96 minutes per epoch. 16 windows is enough to
# rank checkpoints for early stopping.
set -euo pipefail
cd "$(dirname "$0")/.."
OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \
  .venv/bin/python scripts/train_baseline.py \
  --epochs 60 --batch-size 16 --train-stride 1 \
  --eval-samples 50 --val-windows 16 \
  2>&1 | tee results/train_full.log
