#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting 4 Next-Best 2.0 EFLOP Jobs on partition=main"
echo " Strict Last-Layer Evaluation Protocol (No Intermediate Layers)"
echo " Cluster Guardrails: cpus-per-task=16, mem=48G, requeue enabled"
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
    job_id=$(sbatch -p main -J "${job_name}" --time="${slurm_time}" "${S_TRAIN}" "${run_name}" "${seed}" "${epochs}" "${warmup}" "${time_limit}" "$@" | awk '{print $NF}')
    echo "Submitted ${job_name} (${run_name}): Job ID ${job_id}"
}

# 1. w2e-47-decsub2: 3D Sub-Patch Unshuffle Decoder (decoder_sub=2) on SOTA w1024-d8-v384 (29 ep)
submit_2e "w2e-47-decsub2" "2.0e/w1024-d8-v384-dp05-g50-fg025-decsub2-nbr" 7338 29 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 model_kwargs.decoder_sub=2 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 2. w2e-48-pyrbmask: Multi-Scale 3D Spatial Pyramid Readout (1^3+3^3+5^3) + Pure Brain Mask Global Pooling (30 ep)
submit_2e "w2e-48-pyrbmask" "2.0e/w1024-d8-v384-dp05-g50-fg025-pyr-bmask" 7338 30 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=pyramid \
    model_kwargs.brain_mask_pool=true

# 3. w2e-49-currd14: D=14 Deep Hierarchy + V=256->V=384 Resolution Curriculum (27 ep @ 256 + 6 ep @ 384 = 33 ep)
submit_2e "w2e-49-currd14" "2.0e/curr-d14-v256-to-v384-dp04-g50-fg025-nbr" 7338 33 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=256 \
    cooldown_epochs=6 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=14 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.04 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 4. w2e-50-curr1280: W=1280, D=10 Wide-Deep Stack + V=256->V=384 Resolution Curriculum (20 ep @ 256 + 5 ep @ 384 = 25 ep)
submit_2e "w2e-50-curr1280" "2.0e/curr-w1280-d10-v256-to-v384-dp06-g50-fg025-nbr" 7338 25 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0028 num_visible=256 \
    cooldown_epochs=5 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.06 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " 4 Next-Best 2.0 EFLOP jobs submitted on partition=main!"
echo "=========================================================="
