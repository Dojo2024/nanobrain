#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting 8-Job Suite: 2 Relaunches + 6 New 2.0 EFLOP Frontiers"
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

# ------------------------------------------------------------
# 1. w2e-25-d12dpfg: Relaunch of D=12 + Stochastic Depth + Dual Loss (22 ep)
# ------------------------------------------------------------
submit_2e "w2e-25-d12dpfg" "2.0e/w1024-d12-v384-dp05-grad05-fg025-nbr" 7338 22 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 2. w2e-26-d10dpfg: Relaunch of D=10 + Stochastic Depth + Dual Loss (26 ep)
# ------------------------------------------------------------
submit_2e "w2e-26-d10dpfg" "2.0e/w1024-d10-v384-dp05-grad05-fg025-nbr" 7338 26 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 3. w2e-33-d12-g75-dp: Deep Hierarchy (D=12) with Heavy Boundary Regularization (22 ep)
# Fuses D=12 depth abstraction with grad=0.75 edge sharpening and drop_path=0.05
# ------------------------------------------------------------
submit_2e "w2e-33-d12g75" "2.0e/w1024-d12-v384-dp05-grad075-nbr" 7338 22 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.75 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 4. w2e-34-w1152-dp08: Intermediate Width (1152) with Stronger Regularization (24 ep)
# Width 1152 has 25% higher capacity; drop_path=0.08 prevents feature redundancy
# ------------------------------------------------------------
submit_2e "w2e-34-w1152dp08" "2.0e/w1152-d8-v384-dp08-grad05-fg025-nbr" 7338 24 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.08 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 5. w2e-35-v448dec3-dp: High Spatial Density (86% Masking) + Stochastic Depth (24 ep)
# 448 visible patches with compact 3-layer decoder and drop_path=0.05
# ------------------------------------------------------------
submit_2e "w2e-35-v448dec3dp" "2.0e/w1024-d8-v448-dec3-dp05-g50-nbr" 7338 24 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=3 model_kwargs.decoder_embed_dim=384 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 6. w2e-36-v512-dp: Maximum Resolution Density (84% Masking) + Stochastic Depth (20 ep)
# 512 visible tokens capturing micro-structures with drop_path=0.05
# ------------------------------------------------------------
submit_2e "w2e-36-v512dp" "2.0e/w1024-d8-v512-dp05-grad05-nbr" 7338 20 2 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=512 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 7. w2e-37-w1280-dp: Ultra-Wide Capacity (1280) + Stochastic Depth (18 ep)
# Massive channel width 1280 regularized with drop_path=0.05 across 18 epochs
# ------------------------------------------------------------
submit_2e "w2e-37-w1280dp" "2.0e/w1280-d8-v384-dp05-grad05-fg025-nbr" 7338 18 2 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 8. w2e-38-d10-g75-dp: Balanced Depth=10 + Heavy Boundary Loss + Drop Path (26 ep)
# Combines our top stroke lesion architecture with heavy edge regularization and drop path
# ------------------------------------------------------------
submit_2e "w2e-38-d10g75dp" "2.0e/w1024-d10-v384-dp05-grad075-fg025-nbr" 7338 26 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.75 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " All 8 scaling jobs successfully queued into SLURM!"
echo "=========================================================="
