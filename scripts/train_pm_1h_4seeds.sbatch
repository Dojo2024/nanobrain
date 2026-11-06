#!/usr/bin/env bash
#
#   Usage: sbatch train_pm_1h_4seeds.sbatch [SEED]
#   Submit one job per seed, e.g.: for s in 7338 1 2 3; do sbatch train_pm_1h_4seeds.sbatch $s; done
#
#   Visible patches ramp 384 -> 640 over the run (matches the repo's own
#   cooldown direction: harder early, more visible late). num_predict fixed
#   at 640. WSD schedule calibrated for ~8k optimizer steps (1 h).
#
#SBATCH --job-name=nb-v3-pm-1h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --output=slurms/pm1h_%j.out
#SBATCH --error=slurms/pm1h_%j.err

set -euo pipefail

cd /admin/home/dojo/nanobrain-v3

export BRAINMARKS_SMRI_DATA="${BRAINMARKS_SMRI_DATA:-data}"

seed="${1:-7338}"
NAME="v3pm"
OUT_DIR="output"
RUN_TAG="progressive_masking_1h"

RUN_NAME="${NAME}_${RUN_TAG}_s${seed}"
RUN_DIR="${OUT_DIR}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

    status=0
    timeout --kill-after=2m "1h" \
    uv run --no-sync python -m nanobrain.train \
        --overrides \
        name="${RUN_NAME}" \
        output_dir="${OUT_DIR}" \
        seed="${seed}" \
        batch_size=32 accum_iter=1 \
        base_lr=0.003 \
        optimizer=muon muon_impl=fused \
        weight_decay=0.05 \
        epochs=40 warmup_epochs=4 \
        lr_schedule_type=cosine \
        num_predict=640 \
        visible_ramp_start=384 \
        visible_ramp_end=640 \
        model_kwargs.embed_dim=384 model_kwargs.depth=12 model_kwargs.num_heads=6 \
        model_kwargs.decoder_embed_dim=192 model_kwargs.decoder_depth=4 model_kwargs.decoder_num_heads=3 \
        model_kwargs.class_tokens=0 \
        model_kwargs.drop_path_rate=0.05 \
        model_kwargs.norm_target=true \
        model_kwargs.grad_loss_weight=0.5 \
        model_kwargs.fg_loss_weight=0.25 \
        model_kwargs.dense_neighbor_pool=true \
        model_kwargs.brain_mask_pool=true \
        wandb=false \
        || status=$?
    if [[ $status -ne 0 && $status -ne 124 ]]; then
        echo "seed ${seed} failed with ${status}; stopping" >&2
        exit ${status}
    fi

    uv run --no-sync python -m brainmarks_smri vit_mae_3d all \
        --overrides \
        "imports=[nanobrain.model]" \
        model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
        output_dir="${RUN_DIR}" \
        name=eval

echo "=== seed ${seed} done (train + eval) ==="
