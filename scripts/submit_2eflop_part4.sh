#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting Part 4 (Jobs 25-32): 2.0 EFLOP SOTA Push"
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
# 25. w2e-25-d12-dp-fg: The Ultimate Frontier Synergy (22 epochs)
# Fuses our #1 deep model (D=12, 0.6557) with #1 regularizer (drop_path=0.05, 0.6564) + dual loss
# ------------------------------------------------------------
submit_2e "w2e-25-d12dpfg" "2.0e/w1024-d12-v384-dp05-grad05-fg025-nbr" 7338 22 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 26. w2e-26-d10-dp-fg: Depth=10 Aspect Ratio + Stochastic Depth (26 epochs)
# Fuses top stroke-lesion model (D=10, 0.6530) with drop_path=0.05
# ------------------------------------------------------------
submit_2e "w2e-26-d10dpfg" "2.0e/w1024-d10-v384-dp05-grad05-fg025-nbr" 7338 26 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 27. w2e-27-d8-dp08-fg: Stronger Stochastic Depth Regularization on D=8 (30 epochs)
# Tests drop_path=0.08 vs 0.05 on champion D=8 backbone
# ------------------------------------------------------------
submit_2e "w2e-27-dp08fg" "2.0e/w1024-d8-v384-dp08-grad05-fg025-nbr" 7338 30 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.08 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 28. w2e-28-d12-dp08-fg: Deeper Architecture with Strong Stochastic Depth (22 epochs)
# 12 layers benefits from stronger drop_path=0.08 to prevent co-adaptation
# ------------------------------------------------------------
submit_2e "w2e-28-d12dp08" "2.0e/w1024-d12-v384-dp08-grad05-fg025-nbr" 7338 22 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.08 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 29. w2e-29-w1152-dp-fg: Intermediate Width Frontier + Stochastic Depth (24 epochs)
# Width 1152 channel representation with drop_path=0.05 and dual loss
# ------------------------------------------------------------
submit_2e "w2e-29-w1152dp" "2.0e/w1152-d8-v384-dp05-grad05-fg025-nbr" 7338 24 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 30. w2e-30-curr-dp-fg: Multi-Stage Resolution Curriculum + Stochastic Depth (28 epochs)
# Combines resolution curriculum (0.6537) with drop_path=0.05
# ------------------------------------------------------------
submit_2e "w2e-30-currdp" "2.0e/curr-v224-to-v448-dp05-g50-fg025-nbr" 7338 28 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 31. w2e-31-d8-dp-g75-fg: Heavy Boundary Regularization + Stochastic Depth (30 epochs)
# Fuses drop_path=0.05 with grad=0.75 (which gave 0.8964 Age R2) + fg=0.25
# ------------------------------------------------------------
submit_2e "w2e-31-g75dp" "2.0e/w1024-d8-v384-dp05-grad075-fg025-nbr" 7338 30 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.75 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 32. w2e-32-v256-dp-fg: High-Sample Masking Compression + Stochastic Depth (36 epochs)
# 92% token masking (V=256, 0.8985 Age R2) with drop_path=0.05 to boost segmentation
# ------------------------------------------------------------
submit_2e "w2e-32-v256dp" "2.0e/w1024-d8-v256-dp05-grad05-fg025-nbr" 7338 36 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " Part 4 (Jobs 25-32) successfully queued into SLURM!"
echo "=========================================================="
