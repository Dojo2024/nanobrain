#!/usr/bin/env bash
#SBATCH --job-name=nb-probe-speed
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:15:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

set -euo pipefail
cd /admin/home/dojo/nanobrain

set -a
source .env 2>/dev/null || true
set +a
export VIRTUAL_ENV=

DATA_ROOT="/data/nanobrain/FOMO300K_images"
OUT_DIR="/data/dojo/nanobrain/output"
NAME="speed_probe/s7338"
RUN_DIR="${OUT_DIR}/${NAME}"
rm -rf "${RUN_DIR}"

echo "Starting 5-minute speed probe on $(hostname)..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

status=0
timeout --kill-after=1m 6m \
    .venv/bin/python -m nanobrain.train \
    --overrides \
    name="${NAME}" \
    output_dir="${OUT_DIR}" \
    seed=7338 \
    data_root="${DATA_ROOT}" \
    train_filelist="${DATA_ROOT}/filelist_subsampled.txt" \
    plot_period=0 \
    wandb=false \
    print_freq=20 \
    "$@" || status=$?

if [[ $status -ne 0 && $status -ne 124 ]]; then
    echo "Training probe failed with status $status" >&2
    exit $status
fi

echo "Probe completed."
