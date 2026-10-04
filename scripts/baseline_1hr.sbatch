#!/usr/bin/env bash
#SBATCH --job-name=baseline
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=main
#SBATCH --account=idle
#SBATCH --qos=low
#SBATCH --no-requeue
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# Baseline seed replicate of scripts/train_1hr.sbatch: default ViT-S MAE trained for a fixed
# 1 h wall time budget, then the default nanobrain eval on checkpoint-last (~10 min).
# usage: sbatch scripts/baseline_1hr.sbatch <seed>
#
# --no-requeue: a resumed run would train past the 1 h budget, so a preempted seed stops
# instead, and the guard below refuses to reuse a run dir that already has a checkpoint.

set -euo pipefail

SEED="$1"
cd /admin/home/dojo/nanobrain

set -a
source .env
set +a
export VIRTUAL_ENV=

DATA_ROOT="/data/nanobrain/FOMO300K_images"
OUT_DIR="/data/dojo/nanobrain/output"
NAME="baseline/s${SEED}"
RUN_DIR="${OUT_DIR}/${NAME}"

if [[ -e "${RUN_DIR}/checkpoint-last.pth" ]]; then
    echo "${RUN_DIR} already has a checkpoint; resuming would break the 1 h budget" >&2
    exit 1
fi

# run training with a fixed wall time budget
# timeout exits 124 when it stops training, which is expected
status=0
timeout --kill-after=2m "1h" \
    .venv/bin/python -m nanobrain.train \
    --overrides \
    name="${NAME}" \
    output_dir="${OUT_DIR}" \
    seed="${SEED}" \
    data_root="${DATA_ROOT}" \
    train_filelist="${DATA_ROOT}/filelist_subsampled.txt" \
    || status=$?
if [[ $status -ne 0 && $status -ne 124 ]]; then
    exit $status
fi

# run evals if successful, ~10 min
.venv/bin/python -m nanobrain.eval.run \
    "${RUN_DIR}/checkpoint-last.pth" \
    "${RUN_DIR}/eval"
