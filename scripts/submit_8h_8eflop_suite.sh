#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

# User preference: Clean, fixed epoch count across all runs
# 70 epochs gives ~6.0 - 8.2 hours of pretraining depending on width/depth
EPOCHS="${1:-70}"
WARMUP=5
TIME_LIMIT="8.5h"
SLURM_TIME="11:00:00"

echo "=========================================================="
echo " Submitting 8-Job Scaling Suite at Fixed ${EPOCHS} Epochs"
echo " Strict Last-Layer Evaluation Protocol (No Intermediate Layers)"
echo " Cluster Guardrails: cpus-per-task=8, mem=48G, requeue enabled"
echo " Full cluster access: nodes n-1 to n-8"
echo "=========================================================="

submit_8h() {
    local job_name="$1"
    local run_name="$2"
    shift 2
    local job_id
    job_id=$(sbatch -J "${job_name}" --time="${SLURM_TIME}" "${S_TRAIN}" "${run_name}" 7338 "${EPOCHS}" "${WARMUP}" "${TIME_LIMIT}" "$@" | awk '{print $NF}')
    echo "Submitted ${job_name} (${run_name}) @ ${EPOCHS} ep: Job ID ${job_id}"
}

# ------------------------------------------------------------
# 1. w8h-01-fg-nbr: The 1.0 EFLOP Champion Scaled to Fixed Epochs
# w1024-d8, Muon lr=3.5e-3, grad=0.5, fg=0.25, dense neighbor pooling
# ------------------------------------------------------------
submit_8h "w8h-01-fg-nbr" "8h/w1024-d8-v384-grad05-fg025-nbr-e${EPOCHS}" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 2. w8h-02-grad-nbr: Gradient Boundary Loss + Dense Neighbor Pooling
# w1024-d8, Muon lr=3.5e-3, grad=0.5, dense neighbor pooling
# ------------------------------------------------------------
submit_8h "w8h-02-grad-nbr" "8h/w1024-d8-v384-grad05-nbr-e${EPOCHS}" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 3. w8h-03-w1152-d8: Wide Channel Frontier
# w1152-d8, Muon lr=3.2e-3, grad=0.5, dense neighbor pooling
# ------------------------------------------------------------
submit_8h "w8h-03-w1152d8" "8h/w1152-d8-v384-grad05-nbr-e${EPOCHS}" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 4. w8h-04-w1024-d12: Deeper Capacity Frontier
# w1024-d12, Muon lr=3.0e-3, grad=0.5, dense neighbor pooling
# ------------------------------------------------------------
submit_8h "w8h-04-w1024d12" "8h/w1024-d12-v384-grad05-nbr-e${EPOCHS}" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 5. w8h-05-v448-dec3: High Patch Density (86% Masking) + Shallow Decoder
# w1024-d8, V=448, Dec=3 (dim=384), grad=0.5, dense neighbor pooling
# ------------------------------------------------------------
submit_8h "w8h-05-v448dec3" "8h/w1024-d8-v448-dec3-g50-nbr-e${EPOCHS}" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=3 model_kwargs.decoder_embed_dim=384 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 6. w8h-06-v512-dense: Maximum Resolution Density (84% Masking)
# w1024-d8, V=512, grad=0.5, dense neighbor pooling
# ------------------------------------------------------------
submit_8h "w8h-06-v512" "8h/w1024-d8-v512-grad05-nbr-e${EPOCHS}" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=512 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 7. w8h-07-curr-res: Resolution Curriculum (V=256 -> V=448)
# w1024-d8, V=448, grad=0.5, dense neighbor pooling
# ------------------------------------------------------------
submit_8h "w8h-07-curr" "8h/curr-v256-to-v448-lr35-g50-nbr-e${EPOCHS}" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 8. w8h-08-w1280-d8: Ultra-Wide Capacity Frontier
# w1280-d8, Muon lr=3.0e-3, grad=0.5, dense neighbor pooling
# ------------------------------------------------------------
submit_8h "w8h-08-w1280d8" "8h/w1280-d8-v384-grad05-nbr-e${EPOCHS}" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " All 8 Scaling pretraining jobs successfully queued into SLURM!"
echo "=========================================================="
