#!/usr/bin/env bash
#SBATCH --job-name=nb-loader-bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=0:45:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# Loader benchmark for the 16 mm runs (PROJECT_STATE.md section 5: they are loader-bound at ~300 vol/s
# with the GPU 44-60% busy). 5 min of training per worker count on the current best 16 mm recipe
# (ps16/w1024-sub2), inside the same 16-CPU allocation as arm_1hr.sbatch.
# Reports, per arm: vol/s and data wait averaged after the first 2 min, GPU util, GPU and host memory.
# usage: sbatch experiments/dojo_1h/loader/bench_workers.sbatch

set -euo pipefail
cd /admin/home/dojo/nanobrain
set -a
source .env
set +a
export VIRTUAL_ENV=

DATA_ROOT="/data/nanobrain/FOMO300K_images"
OUT_DIR="/data/dojo/nanobrain/output"
BENCH="${OUT_DIR}/loader-bench"
BASE="base_lr=0.003 optimizer=muon muon_impl=batched norm_target=true model_kwargs.embed_dim=1024 model_kwargs.num_heads=16 patch_size=16 num_visible=168 num_predict=168 model_kwargs.decoder_sub=2 model_kwargs.decoder_sub_depth=1"
mkdir -p "${BENCH}"
echo "node $(hostname -s), $(nproc) CPUs visible, MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB"

for nw in 8 12 15; do
    arm="nw${nw}"
    echo "-- ${arm}"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 10 > "${BENCH}/gpu-${arm}.csv" &
    smi=$!
    status=0
    timeout --kill-after=1m 5m .venv/bin/python -m nanobrain.train --overrides \
        name="loader-bench/${arm}" output_dir="${OUT_DIR}" seed=0 print_freq=50 \
        data_root="${DATA_ROOT}" train_filelist="${DATA_ROOT}/filelist_subsampled.txt" \
        ${BASE} num_workers=${nw} > "${BENCH}/train-${arm}.log" 2>&1 || status=$?
    kill "${smi}" || true
    if [[ $status -ne 0 && $status -ne 124 ]]; then
        echo "${arm} FAILED (exit ${status})"; tail -5 "${BENCH}/train-${arm}.log"
        continue
    fi
    # log lines every ~5 s; skip the first 24 (~2 min: startup, compile, worker warmup)
    grep "vol/s" "${BENCH}/train-${arm}.log" | sed -E 's/.*vol\/s: ([0-9]+).*data: ([0-9.]+).*max mem: ([0-9]+)MB.*/\1 \2 \3/' |
        awk 'NR > 24 {v += $1; d += $2; n++; m = $3} END {printf "vol/s %.0f, data wait %.3f s/step, GPU max mem %.1f GB (%d lines)\n", v / n, d / n, m / 1024, n}'
    awk -F', ' 'NR > 12 {u += $1; n++} END {printf "GPU util mean %.0f%%\n", u / n}' "${BENCH}/gpu-${arm}.csv"
    echo "MemAvailable after: $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB"
done
echo "== done"
