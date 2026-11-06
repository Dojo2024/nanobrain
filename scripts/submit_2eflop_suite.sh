#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting 4-Job Suite: 2.0 EFLOP / ~2-Hour Nanobrain Scaling"
echo " Strict Last-Layer Evaluation Protocol (No Intermediate Layers)"
echo " Cluster Guardrails: cpus-per-task=8, mem=48G, requeue enabled"
echo " Full cluster access: nodes n-1 to n-8"
echo "=========================================================="

submit_2e() {
    local job_name="$1"
    local run_name="$2"
    local seed="$3"
    local epochs="$4"
    local warmup="$5"
    local time_limit="$6"
    local slurm_time="$7"
    shift 7
    local job_id
    job_id=$(sbatch -J "${job_name}" --time="${slurm_time}" "${S_TRAIN}" "${run_name}" "${seed}" "${epochs}" "${warmup}" "${time_limit}" "$@" | awk '{print $NF}')
    echo "Submitted ${job_name} (${run_name}): Job ID ${job_id}"
}

# ------------------------------------------------------------
# 1. w2e-01-fg-nbr: Direct 2.0E Scale of 1.0E Champion (0.6482 -> ~0.658)
# w1024-d8, 30 ep @ V=384, Muon 3.5e-3, grad=0.5, fg=0.25, dense neighbor pool
# ------------------------------------------------------------
submit_2e "w2e-01-fg-nbr" "2.0e/w1024-d8-v384-grad05-fg025-nbr" 7338 30 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 2. w2e-02-v256-fg-nbr: High-Sample 92% Masking Compression (36 epochs)
# w1024-d8, 36 ep @ V=256, Muon 3.5e-3, grad=0.5, fg=0.25, dense neighbor pool
# ------------------------------------------------------------
submit_2e "w2e-02-v256-nbr" "2.0e/w1024-d8-v256-grad05-fg025-nbr" 7338 36 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 3. w2e-03-w1152-nbr: Intermediate Width Sweet-Spot (24 epochs)
# w1152-d8, 24 ep @ V=384, Muon 3.2e-3, grad=0.5, dense neighbor pool
# ------------------------------------------------------------
submit_2e "w2e-03-w1152" "2.0e/w1152-d8-v384-grad05-nbr" 7338 24 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 4. w2e-04-v448dec3-nbr: High Spatial Density + Compact Decoder (24 epochs)
# w1024-d8, 24 ep @ V=448, Dec=3 (dim=384), Muon 3.5e-3, grad=0.5, dense neighbor pool
# ------------------------------------------------------------
submit_2e "w2e-04-v448dec3" "2.0e/w1024-d8-v448-dec3-g50-nbr" 7338 24 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=3 model_kwargs.decoder_embed_dim=384 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " All 4 2.0 EFLOP pretraining jobs successfully queued into SLURM!"
echo "=========================================================="
