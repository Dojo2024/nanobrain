#!/usr/bin/env bash
#SBATCH --job-name=nb-siam-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# SiamJEPA GPU smoke test (bf16 + torch.compile + EMA teacher, never run on GPU): one epoch each of
# the planned 16 mm arm (default width 384) without and with Random Shuffle Teacher. Reports vol/s (MAE p16
# runs ~265-335, loader-bound), GPU util and memory, and the loss trend.
# usage: sbatch experiments/dojo_1h/siamjepa/smoke.sbatch

set -euo pipefail
cd /admin/home/dojo/nanobrain
set -a
source .env
set +a
export VIRTUAL_ENV=

DATA_ROOT="/data/nanobrain/FOMO300K_images"
OUT_DIR="/data/dojo/nanobrain/output"
SMOKE="${OUT_DIR}/siamjepa-smoke"
mkdir -p "${SMOKE}"
BASE="base_lr=0.003 optimizer=muon muon_impl=batched weight_decay=0.1 patch_size=16 num_visible=168 num_predict=168 model_kwargs.objective=siamjepa"

for arm in plain rst; do
    extra=""
    [[ "${arm}" == rst ]] && extra="rst_min=60"
    echo "-- ${arm}"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 10 > "${SMOKE}/gpu-${arm}.csv" &
    smi=$!
    .venv/bin/python -m nanobrain.train --overrides \
        name="siamjepa-smoke/${arm}" output_dir="${OUT_DIR}" seed=0 print_freq=100 stop_epoch=1 \
        data_root="${DATA_ROOT}" train_filelist="${DATA_ROOT}/filelist_subsampled.txt" \
        ${BASE} ${extra} > "${SMOKE}/train-${arm}.log" 2>&1 || echo "${arm} FAILED"
    kill "${smi}" || true
    grep "vol/s" "${SMOKE}/train-${arm}.log" | sed -n '1p;4p;8p;12p;$p' | cut -c1-200 || tail -5 "${SMOKE}/train-${arm}.log"
    awk -F', ' 'NR > 6 {u += $1; m = ($2 > m ? $2 : m); n++} END {printf "GPU util mean %.0f%%, peak mem %.1f GB\n", u / n, m / 1024}' "${SMOKE}/gpu-${arm}.csv"
done
echo "== done"
