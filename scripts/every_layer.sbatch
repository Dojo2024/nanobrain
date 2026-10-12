#!/usr/bin/env bash
#SBATCH --job-name=every-layer
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --exclude=n-1,n-2,n-3,n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# Single-layer probes at every encoder block not yet scored, for several runs at one seed, on one
# GPU. Ordered by layer, so each finished layer is comparable across the runs. Each probe is one
# readout_eval.sbatch pass (~7.3 min, 8 tasks). Same footprint as layers_one_gpu.sbatch: peak host
# RAM ~14 GB, GPU memory <= 31 GB, one step at a time.
# usage: sbatch -J nb-l1-every-layer scripts/every_layer.sbatch <seed> "<layers>" <run> [<run>...]
#   e.g. ... 7338 "1 2 4 5 7 8 10 11" a17/batched-nt l1/dhc4-w384-matched l1/mhc4-w384

set -euo pipefail
SEED="$1"
LAYERS="$2"
shift 2
cd /admin/home/dojo/nanobrain

for layer in ${LAYERS}; do
    for run in "$@"; do
        bash scripts/readout_eval.sbatch "${run}" "${SEED}" "l${layer}" "" --layers "${layer}"
    done
done
echo "[$(date +%T)] all done"
