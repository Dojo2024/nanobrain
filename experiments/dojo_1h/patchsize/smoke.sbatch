#!/usr/bin/env bash
#SBATCH --job-name=nb-ps-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-1,n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# Patch-size smoke test (nanobrain-ideas.md, patch size 12/16):
# 1. in-head patch counts at patch 8/12/16, to scale num_visible/num_predict
# 2. 5 min of training per arm on the width 512 Faster Muon recipe, for vol/s and GPU use
# 3. the frozen FOMO eval and the brainmarks alignment check on random-init patch 12/16 models
# usage: sbatch experiments/dojo_1h/patchsize/smoke.sbatch

set -euo pipefail
cd /admin/home/dojo/nanobrain
set -a
source .env
set +a
export VIRTUAL_ENV=

DATA_ROOT="/data/nanobrain/FOMO300K_images"
OUT_DIR="/data/dojo/nanobrain/output"
SMOKE="${OUT_DIR}/patchsize-smoke"
BASE="base_lr=0.003 optimizer=muon muon_impl=batched norm_target=true model_kwargs.embed_dim=512 model_kwargs.num_heads=8"
mkdir -p "${SMOKE}"

echo "== 1. in-head patch counts"
.venv/bin/python experiments/dojo_1h/patchsize/smoke.py count 300

echo "== 2. 5 min of training per arm"
declare -A ARMS=(
    [p8]="patch_size=8"
    [p12]="patch_size=12 num_visible=384 num_predict=384"
    [p16]="patch_size=16 num_visible=160 num_predict=160"
    [p16-w768]="patch_size=16 num_visible=160 num_predict=160 model_kwargs.embed_dim=768 model_kwargs.num_heads=12"
    [p16-ns8]="patch_size=16 num_visible=160 num_predict=160 num_samples=8"
)
for arm in p8 p12 p16 p16-w768 p16-ns8; do
    echo "-- ${arm}: ${ARMS[$arm]}"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 10 > "${SMOKE}/gpu-${arm}.csv" &
    smi=$!
    status=0
    timeout --kill-after=1m 5m .venv/bin/python -m nanobrain.train --overrides \
        name="patchsize-smoke/${arm}" output_dir="${OUT_DIR}" seed=0 print_freq=50 \
        data_root="${DATA_ROOT}" train_filelist="${DATA_ROOT}/filelist_subsampled.txt" \
        ${BASE} ${ARMS[$arm]} > "${SMOKE}/train-${arm}.log" 2>&1 || status=$?
    kill "${smi}" || true
    if [[ $status -ne 0 && $status -ne 124 ]]; then
        echo "${arm} FAILED (exit ${status})"; tail -5 "${SMOKE}/train-${arm}.log"
        continue
    fi
    echo "${arm}: last log line with vol/s:"
    grep "vol/s" "${SMOKE}/train-${arm}.log" | tail -1
    # skip the first minute (warmup, compile) in the GPU average
    awk -F', ' 'NR > 6 {u += $1; m = ($2 > m ? $2 : m); n++} END {printf "GPU util mean %.0f%%, peak mem %.1f GB\n", u / n, m / 1024}' "${SMOKE}/gpu-${arm}.csv"
done

echo "== 3. eval shape checks on random-init patch 12/16 models"
for p in 12 16; do
    ckpt="${SMOKE}/init-p${p}/checkpoint-init.pth"
    .venv/bin/python experiments/dojo_1h/patchsize/smoke.py init "${ckpt}" ${BASE} patch_size=${p}
    .venv/bin/python -m nanobrain.eval.run "${ckpt}" "${SMOKE}/init-p${p}/eval" || echo "FOMO eval p${p} FAILED"
    DATA_ROOT=/data/connor/brain_datasets/datasets \
    PYTHONPATH=/admin/home/dojo/brainmarks-smri/src:/admin/home/dojo/nanobrain/experiments/brainmarks \
    OMP_NUM_THREADS=2 \
        .venv/bin/python experiments/brainmarks/check_alignment.py "${ckpt}" || echo "brainmarks check p${p} FAILED"
done
echo "== done"
