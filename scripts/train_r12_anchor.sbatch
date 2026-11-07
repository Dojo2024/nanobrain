#!/usr/bin/env bash
#
#   Anchor recreation of the current labless leader: r12_nat_lvl_mf
#   (run_sub_8eb1eb614b, open 0.5334) using Mihir's published run source,
#   which carries the level-set head not present in any git branch.
#   One seed validates the environment and anchors the baseline we build on.
#
#   Usage: sbatch train_r12_anchor.sbatch [SEED]
#
#SBATCH --job-name=nb-r12
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --output=slurms/r12_%j.out
#SBATCH --error=slurms/r12_%j.err

set -euo pipefail

cd /admin/home/dojo/nanobrain-r12

export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export OMP_NUM_THREADS=2

SEED="${1:-7338}"
NAME="native-r12-s${SEED}"
RUN_DIR="output/${NAME}"

uv run --no-sync python -m nanobrain.train \
    --config src/nanobrain/config/r12_nat_lvl_mf.yaml \
    --overrides name="${NAME}" seed="${SEED}"

uv run --no-sync python -m brainmarks_smri vit_mae_3d open \
    --overrides \
    "imports=[nanobrain.model]" \
    model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
    output_dir="${RUN_DIR}" \
    name=eval

echo "=== native-r12 s${SEED} done (submit: uv run python scripts/submit.py ${RUN_DIR}) ==="
