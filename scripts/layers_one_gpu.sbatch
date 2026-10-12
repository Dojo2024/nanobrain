#!/usr/bin/env bash
#SBATCH --job-name=layers
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:30:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --exclude=n-1,n-2,n-3,n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# Every per-layer measurement for one run on a single GPU, seeds one after another:
# layer similarity (scripts/layer_similarity.py, all 12 blocks, ~1 min), then per seed the
# layer 3, 6, 9 probes and the dense half of the split readout (cat6-9-12 on the 6 dense tasks),
# via scripts/readout_eval.sbatch run as a plain script. Same readouts as l1/dhc4-w384-matched
# and a17/batched-nt. ~25 min per seed.
# Measured on the same readouts (518023, 518036): peak host RAM 12.6 and 13.8 GB, peak GPU memory
# 11.8 and 31.3 GB; similarity 2.7 GB RAM. Steps run one at a time, so --mem=32G is over 2x the
# peak. n-1..n-4 excluded on 2026-10-07 (13-40 GB host RAM free, or draining).
# usage: sbatch -J nb-l1-mhc4-layers scripts/layers_one_gpu.sbatch l1/mhc4-w384 7338 1 2 3

set -euo pipefail
RUN="$1"
shift
cd /admin/home/dojo/nanobrain
set -a
source .env
set +a
export VIRTUAL_ENV=

DENSE="fomo_task1_adc_lesion fomo_task2_dwi_tumor fomo_task3_t1w_hippocampus fomo_task4_t2w_nerve fomo_task4_t2w_vessel fomo_task5_t1w_cortex"

.venv/bin/python scripts/layer_similarity.py --runs "${RUN}" --seeds "$@"
for seed in "$@"; do
    bash scripts/readout_eval.sbatch "${RUN}" "${seed}"
    bash scripts/readout_eval.sbatch "${RUN}" "${seed}" cat6-9-12 "${DENSE}" --layers 6 9 12
done
echo "[$(date +%T)] all done"
