#!/usr/bin/env bash
#
#   Eval-only fill: runs just the sweep's eval stage against a finished arm's
#   checkpoint, writing eval/aggregate.json under the arm's RUN_DIR. Use for
#   arms whose train completed but whose eval died (e.g. the pre-fix eval).
#
#SBATCH --job-name=nb-iso-eval
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --time=00:30:00
#SBATCH --output=slurms/eval_%j.out
#SBATCH --error=slurms/eval_%j.err

set -euo pipefail
cd /admin/home/dojo/nanobrain-sweep

RUN_DIR="${1:?output dir of the arm, e.g. iso/base-e768-d12/base-e768-d12-s7338}"
mkdir -p "${RUN_DIR}/eval" slurms

export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export OMP_NUM_THREADS=2
export RESULT_DIR="${RUN_DIR}/eval"

uv run --no-sync python -m brainmarks_smri vit_mae_3d open \
    --overrides \
    "imports=[nanobrain.model]" \
    model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
    output_dir="${RUN_DIR}/eval" \
    name=eval

echo "=== eval done: ${RUN_DIR}/eval ==="
