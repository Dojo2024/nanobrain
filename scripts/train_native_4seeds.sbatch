#!/usr/bin/env bash
#
#   Arm #2: native-resolution replication (Mihir's vitb_p8_bs64_native, exact
#   config from Labless run_sub_e8ef2f0721), 4 seeds. No auto-submit.
#
#   Usage: sbatch train_native_4seeds.sbatch [SEED]
#
#SBATCH --job-name=nb-native
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --output=slurms/native_%j.out
#SBATCH --error=slurms/native_%j.err

set -euo pipefail

cd /admin/home/dojo/nanobrain-native

export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export OMP_NUM_THREADS=2

SEED="${1:-7338}"
NAME="native-vitb-s${SEED}"
RUN_DIR="output/${NAME}"

# Train: exact Labless config; seed + name overridden per job.
uv run --no-sync python -m nanobrain.train \
    --config src/nanobrain/config/vitb_p8_bs64_native.yaml \
    --overrides name="${NAME}" seed="${SEED}"

# Eval: Open-11 group, pinned harness (8ef7115 per this repo's lock).
uv run --no-sync python -m brainmarks_smri vit_mae_3d open \
    --overrides \
    "imports=[nanobrain.model]" \
    model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
    output_dir="${RUN_DIR}" \
    name=eval

echo "=== native s${SEED} done (submit manually: uv run python scripts/submit.py ${RUN_DIR}) ==="
