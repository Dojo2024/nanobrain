#!/usr/bin/env bash
#SBATCH --job-name=nb-reeval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:15:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

set -euo pipefail

RUN_REL="$1"

cd /admin/home/dojo/nanobrain
export VIRTUAL_ENV=
export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export PYTHONPATH=/admin/home/dojo/brainmarks-smri/src:/admin/home/dojo/nanobrain/src
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="/data/dojo/nanobrain/output/isoflop/${RUN_REL}"

if [[ ! -f "${RUN_DIR}/checkpoint-last.pth" ]]; then
    echo "ERROR: ${RUN_DIR}/checkpoint-last.pth not found" >&2
    exit 1
fi

# Preserve legacy 19-task v1 evaluation if present and lacking aggregate.json
if [[ -d "${RUN_DIR}/brainmarks" && ! -f "${RUN_DIR}/brainmarks/aggregate.json" ]]; then
    if [[ ! -d "${RUN_DIR}/brainmarks_v1_19tasks" ]]; then
        mv "${RUN_DIR}/brainmarks" "${RUN_DIR}/brainmarks_v1_19tasks"
    else
        rm -rf "${RUN_DIR}/brainmarks"
    fi
fi

# Rule for nanobrain: evaluate ONLY on the last layer (no intermediate layers)
.venv/bin/python -m brainmarks_smri vit_mae_3d all \
    --overrides \
    "imports=[nanobrain.model]" \
    model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
    output_dir="${RUN_DIR}" \
    name=brainmarks

