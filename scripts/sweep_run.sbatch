#!/usr/bin/env bash
#
#   Generic sweep runner: one training run per (config, seed). The config path is
#   relative to configs/ (e.g. "iso/base-e768-d12" or "combine/<winner>-rampdn"),
#   carries its own total_steps (iso-flop budget) and time_limit. Train is step
#   based, same accounting as every other run on the board; the labless submit
#   happens separately once the run's eval aggregate is in.
#
#SBATCH --job-name=nb-iso
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --output=slurms/iso_%j.out
#SBATCH --error=slurms/iso_%j.err

set -euo pipefail

cd /admin/home/dojo/nanobrain-sweep

CONFIG_REL="${1:?config path relative to configs/, e.g. iso/base-e768-d12}"
SEED="${2:?seed}"
NAME="$(basename "${CONFIG_REL}")-s${SEED}"
RUN_DIR="output/${CONFIG_REL}/s${SEED}"
mkdir -p "${RUN_DIR}" slurms

export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export OMP_NUM_THREADS=2

# train to the config's total_steps (iso-flop budget); time_limit caps the wall
timeout --kill-after=2m "$(python3 -c "import yaml; print(yaml.safe_load(open('configs/${CONFIG_REL}.yaml'))['time_limit'])")m" \
    .venv/bin/python -m nanobrain.train \
    --config "configs/${CONFIG_REL}.yaml" \
    --overrides name="${NAME}" output_dir="output/${CONFIG_REL}" seed="${SEED}" \
    || status=$?

status=${status:-0}
if [[ $status -ne 0 && $status -ne 124 ]]; then
    echo "training failed with ${status}" >&2
    exit "${status}"
fi

# eval: open group (the 11 labless-scored tasks), final layer readout
uv run --no-sync python -m brainmarks_smri vit_mae_3d open \
    --overrides \
    "imports=[nanobrain.model]" \
    model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
    model_kwargs.readout_layers=null \
    output_dir="${RUN_DIR}" \
    name=eval

echo "=== iso ${CONFIG_REL} s${SEED} done (train status ${status}) ==="
