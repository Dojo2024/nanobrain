#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Resubmitting 4 Cancelled 2.0 EFLOP Jobs from Scratch"
echo " Strict Last-Layer Evaluation Protocol (No Intermediate Layers)"
echo " Cluster Guardrails: cpus-per-task=16, mem=48G, requeue enabled"
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

# 1. w2e-39-curr256-384dp: Resolution Curriculum (V=256 -> V=384) + Stochastic Depth (26 ep)
submit_2e "w2e-39-curr256dp" "2.0e/curr-v256-to-v384-dp05-g50-fg025-nbr" 7338 26 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    cooldown_epochs=6 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 2. w2e-40-d10v256dp: Deeper Hierarchy (D=10) with Compact Patch Resolution (V=256) (26 ep)
submit_2e "w2e-40-d10v256dp" "2.0e/w1024-d10-v256-dp05-grad05-fg025-nbr" 7338 26 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 3. w2e-41-d12dp03fg: Deep Hierarchy (D=12) with Gentle Stochastic Depth (dp=0.03) (22 ep)
submit_2e "w2e-41-d12dp03fg" "2.0e/w1024-d12-v384-dp03-grad05-fg025-nbr" 7338 22 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.03 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 4. w2e-43-d8g60fg30: SOTA Champion Fine-Tuning (grad=0.60, fg=0.30) (30 ep)
submit_2e "w2e-43-d8g60fg30" "2.0e/w1024-d8-v384-dp05-grad060-fg030-nbr" 7338 30 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.60 \
    model_kwargs.fg_loss_weight=0.30 model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " 4 Cancelled jobs successfully resubmitted into SLURM!"
echo "=========================================================="
