#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting 8 Frontier 8.0 EFLOP Scaling Jobs"
echo " Strict Last-Layer Evaluation Protocol (No Intermediate Layers)"
echo " Cluster Guardrails: cpus-per-task=16, mem=48G, requeue enabled"
echo " Full cluster access: nodes n-1 to n-8"
echo "=========================================================="

submit_8e() {
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

TIME_LIMIT="11.0h"
SLURM_TIME="14:00:00"

# ------------------------------------------------------------
# 1. w8e-01-d8dp05: The #1 All-19 SOTA Champion Scaled to 8.0 EFLOP (120 ep)
# Source 2.0E: 0.6564 All-19, 0.7059 Core-5, 0.8950 Age-2, 0.7922 TC Dice
# ------------------------------------------------------------
submit_8e "w8e-01-d8dp05" "8.0e/w1024-d8-v384-dp05-grad05-fg025-nbr" 7338 120 8 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 2. w8e-02-d10dp05: Tumor Core SOTA Record Champion at 8.0 EFLOP (104 ep)
# Source 2.0E: 0.8029 Tumor Core Dice (All-Time Record), 0.6533 All-19, 0.7075 Core-5
# ------------------------------------------------------------
submit_8e "w8e-02-d10dp05" "8.0e/w1024-d10-v384-dp05-grad05-fg025-nbr" 7338 104 7 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 3. w8e-03-d12nbr: The #2 All-19 Champion & Deep Backbone Frontier at 8.0 EFLOP (88 ep)
# Source 2.0E: 0.6557 All-19, 0.7088 Core-5, 0.8022 TC Dice
# ------------------------------------------------------------
submit_8e "w8e-03-d12nbr" "8.0e/w1024-d12-v384-grad05-nbr" 7338 88 6 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 4. w8e-04-d12dp05: Deep Regularized Hierarchy at 8.0 EFLOP (88 ep)
# Source 2.0E: 0.6532 All-19, 0.7047 Core-5, 0.8944 Age-2, 0.7940 TC Dice
# ------------------------------------------------------------
submit_8e "w8e-04-d12dp05" "8.0e/w1024-d12-v384-dp05-grad05-fg025-nbr" 7338 88 6 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 5. w8e-05-d8fg025: Core-5 Segmentation SOTA Champion at 8.0 EFLOP (120 ep)
# Source 2.0E: 0.7091 Core-5 (Top Seg), 0.4709 ATLAS Lesion (All-Time Record), 0.6538 All-19
# ------------------------------------------------------------
submit_8e "w8e-05-d8fg025" "8.0e/w1024-d8-v384-grad05-fg025-nbr" 7338 120 8 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 6. w8e-06-v256dp: Age-2 SOTA Champion / High Token Throughput at 8.0 EFLOP (144 ep)
# Source 2.0E: 0.8994 Age-2 R2, 0.6543 All-19, 92% token masking compression
# ------------------------------------------------------------
submit_8e "w8e-06-v256dp" "8.0e/w1024-d8-v256-dp05-grad05-fg025-nbr" 7338 144 8 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 7. w8e-07-curr448: High-Density Curriculum Frontier at 8.0 EFLOP (116 ep)
# Source 2.0E: 0.6537 All-19, 0.7066 Core-5 (92 ep @ V=256 + 24 ep @ V=448 cooldown)
# ------------------------------------------------------------
submit_8e "w8e-07-curr448" "8.0e/curr-v256-to-v448-dp05-g50-fg025-nbr" 7338 116 8 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    cooldown_epochs=24 cooldown_num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 8. w8e-08-w1280dp: Ultra-Wide Representation Frontier at 8.0 EFLOP (88 ep)
# Source 2.0E: 0.6531 All-19, 0.7028 Core-5, 0.8846 Age-2
# ------------------------------------------------------------
submit_8e "w8e-08-w1280dp" "8.0e/w1280-d8-v384-dp05-grad05-fg025-nbr" 7338 88 6 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " All 8 Frontier 8.0 EFLOP jobs successfully queued into SLURM!"
echo "=========================================================="

