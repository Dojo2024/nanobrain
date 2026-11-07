#!/usr/bin/env bash
#   Eval a w1h run on the 11-task Open group.
#   Usage: sbatch eval_w1h_open.sbatch <RUN_NAME>
#SBATCH --job-name=nb-eval-open
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --partition=main
#SBATCH --account=training

set -euo pipefail
cd /admin/home/dojo/nanobrain-v3

RUN="$1"
RUN_DIR="output/${RUN}"

if [[ ! -f "${RUN_DIR}/checkpoint-last.pth" ]]; then
    echo "checkpoint missing: ${RUN_DIR}" >&2
    exit 1
fi

uv run --no-sync python -m brainmarks_smri vit_mae_3d open \
    --overrides \
    "imports=[nanobrain.model]" \
    model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
    model_kwargs.readout_layers=null \
    output_dir="${RUN_DIR}" \
    name=eval_open
