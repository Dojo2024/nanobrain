#!/usr/bin/env bash
#SBATCH --job-name=nanobrain-sub2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2:30:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --error=slurms/slurm-%j.err

set -euo pipefail

OUT_DIR="output"
NAME="ps16_w1024_sub2"
RUN_DIR="${OUT_DIR}/${NAME}"

# run 1h training with sub2 pixel-unshuffle decoder and BatchedMuon
status=0
timeout --kill-after=2m "1h" \
    uv run --no-sync python -m nanobrain.train \
    --config-name ps16_w1024_sub2 \
    --overrides \
    name="${NAME}" \
    output_dir="${OUT_DIR}" \
    || status=$?
if [[ $status -ne 0 && $status -ne 124 ]]; then
    exit $status
fi

# Run FOMO native eval
uv run --no-sync python -m nanobrain.eval.run \
    "${RUN_DIR}/checkpoint-last.pth" \
    "${RUN_DIR}/eval"
