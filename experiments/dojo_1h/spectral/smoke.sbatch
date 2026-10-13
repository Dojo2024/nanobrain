#!/usr/bin/env bash
#SBATCH --job-name=nb-a2-smoke
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

# A2 spectral loss smoke test on GPU (bf16 autocast + torch.compile with complex FFTs, never run before):
# 1. one epoch (~5 min, the 40-epoch schedule unchanged) of the n11/width512 recipe with spectral_weight 0.1;
#    compare vol/s with the control's ~172-180
# 2. measure.py: spectral term vs masked MSE on real batches, for this checkpoint and for the 1 h control
#    (n11/width512 s7338, trained without the term), to pick the weights of the real arm
# usage: sbatch experiments/dojo_1h/spectral/smoke.sbatch

set -euo pipefail
cd /admin/home/dojo/nanobrain
set -a
source .env
set +a
export VIRTUAL_ENV=

DATA_ROOT="/data/nanobrain/FOMO300K_images"
OUT_DIR="/data/dojo/nanobrain/output"
NAME="spectral-smoke/w0.1"
mkdir -p "${OUT_DIR}/spectral-smoke"

.venv/bin/python -m nanobrain.train --overrides \
    name="${NAME}" output_dir="${OUT_DIR}" seed=0 print_freq=50 stop_epoch=1 \
    data_root="${DATA_ROOT}" train_filelist="${DATA_ROOT}/filelist_subsampled.txt" \
    base_lr=0.003 optimizer=muon muon_impl=batched norm_target=true \
    model_kwargs.embed_dim=512 model_kwargs.num_heads=8 spectral_weight=0.1 \
    > "${OUT_DIR}/${NAME}.log" 2>&1
grep "vol/s" "${OUT_DIR}/${NAME}.log" | sed -n '1p;8p;16p;24p;$p'
grep -i "warn\|graph break\|recompil" "${OUT_DIR}/${NAME}.log" | sort | uniq -c | head -5 || true

.venv/bin/python experiments/dojo_1h/spectral/measure.py \
    "${OUT_DIR}/${NAME}/checkpoint-last.pth" "${OUT_DIR}/n11/width512/s7338/checkpoint-last.pth"
echo "== done"
