#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting 4-8 Hour Scaling Pretraining Jobs"
echo " Strict Last-Layer Evaluation Protocol (No Intermediate Layers)"
echo " Guardrails: cpus-per-task=8, mem=48G, requeue enabled"
echo " Full cluster access: nodes n-1 to n-8"
echo "=========================================================="

submit_scale() {
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
# 1. w7-01-8h-w1024d8: 8-Hour Scaling Champion (8.8 EFLOP)
# 72 epochs @ V=384 with Muon + Gradient Loss
# ------------------------------------------------------------
submit_scale "w7-01-8h-w1024d8" "8.8e/w1024-d8-v384-grad05" 7338 72 6 "8h" "10:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# 2. w7-02-8h-w1024d12: 8-Hour Deep Capacity Scaling (8.8 EFLOP)
# 56 epochs @ V=384 with Muon + Gradient Loss
# ------------------------------------------------------------
submit_scale "w7-02-8h-w1024d12" "8.8e/w1024-d12-v384-grad05" 7338 56 5 "8h" "10:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# 3. w7-03-4h-v448dec3: 4-Hour Ultra-Dense Patch Resolution (4.4 EFLOP)
# 32 epochs @ V=448 with Shallow Decoder (Dec=3, Dim=384) + Gradient Loss
# ------------------------------------------------------------
submit_scale "w7-03-4h-v448dec3" "4.4e/w1024-d8-v448-dec3-g50" 7338 32 4 "4h" "6:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=3 model_kwargs.decoder_embed_dim=384 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# 4. w7-04-6h-w1152d8: 6-Hour Wide Representation Frontier (6.6 EFLOP)
# 45 epochs @ V=384 with Width=1152 + Muon + Gradient Loss
# ------------------------------------------------------------
submit_scale "w7-04-6h-w1152d8" "6.6e/w1152-d8-v384-grad05" 7338 45 4 "6h" "8:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

echo "=========================================================="
echo " All 4 Scaling pretraining jobs successfully queued into SLURM!"
echo "=========================================================="
