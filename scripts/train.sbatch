#!/usr/bin/env bash
#
#   sbatch scripts/train.sbatch NAME [OVERRIDES...]
#
# Smoke test: sbatch scripts/train.sbatch smoke time_limit=5

# Slurm options for the MedARC cluster. Edit for yours.
#SBATCH --job-name=nanobrain
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurms/%j.out

set -euo pipefail

NAME=$1
shift
RUN_DIR="output/${NAME}"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

# Train until total_steps or time_limit (2 hr default).
uv run --no-sync python -m nanobrain.train \
    --overrides output_dir=output name="${NAME}" "$@"

# Evaluate the last checkpoint. Finished tasks are skipped when requeued.
uv run --no-sync python -m brainmarks_smri vit_mae_3d open \
    --overrides \
    "imports=[nanobrain.model]" \
    model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
    output_dir="${RUN_DIR}" \
    name=eval
