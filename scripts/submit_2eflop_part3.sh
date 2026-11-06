#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting Part 3 (Jobs 17-24): 2.0 EFLOP Scaling Suite"
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
# 17. w2e-17-v224-fg-nbr: Ultra-Compression 93% Masking Frontier (40 epochs)
# 40 epochs allows learning extensive data semantics across 3.4M sequence representations
# ------------------------------------------------------------
submit_2e "w2e-17-v224fg" "2.0e/w1024-d8-v224-grad05-fg025-nbr" 7338 40 4 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=224 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 18. w2e-18-lr40-champ: Higher Muon Learning Rate Frontier (30 epochs)
# Muon lr=0.0040 accelerates feature separation in deep representation space
# ------------------------------------------------------------
submit_2e "w2e-18-lr40fg" "2.0e/w1024-d8-v384-lr40-grad05-fg025-nbr" 7338 30 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0040 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 19. w2e-19-warm4-champ: Extended Warmup Stabilization (30 epochs)
# 4 epochs warmup allows smoother initial Muon Newton-Schulz orthogonalization
# ------------------------------------------------------------
submit_2e "w2e-19-warm4" "2.0e/w1024-d8-v384-warmup4-grad05-fg025-nbr" 7338 30 4 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 20. w2e-20-dec2-fg-nbr: Ultra-Shallow 2-Layer Decoder (32 epochs)
# 2-layer decoder forces virtually all spatial reasoning into the encoder backbone
# ------------------------------------------------------------
submit_2e "w2e-20-dec2fg" "2.0e/w1024-d8-v384-dec2-grad05-fg025-nbr" 7338 32 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 21. w2e-21-curr-tri: Tri-Phase Resolution Curriculum (V=224 -> 320 -> 448) (28 epochs)
# Progressive multi-stage resolution refinement
# ------------------------------------------------------------
submit_2e "w2e-21-currtri" "2.0e/curr-v224-to-v448-lr35-g50-fg025-nbr" 7338 28 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 22. w2e-22-w1152-v256: Intermediate Width with High-Sample Masking (28 epochs)
# Combines width 1152 representation power with 92% token masking compression
# ------------------------------------------------------------
submit_2e "w2e-22-w1152v256" "2.0e/w1152-d8-v256-grad05-fg025-nbr" 7338 28 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 23. w2e-23-w1024d10-fg: Balanced Depth=10 Architecture (26 epochs)
# 10 transformer blocks provides an optimal depth/width aspect ratio
# ------------------------------------------------------------
submit_2e "w2e-23-d10fg" "2.0e/w1024-d10-v384-grad05-fg025-nbr" 7338 26 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 24. w2e-24-drop-path: Stochastic Depth Regularization (30 epochs)
# Drop path rate = 0.05 prevents co-adaptation across 30 epochs
# ------------------------------------------------------------
submit_2e "w2e-24-droppath" "2.0e/w1024-d8-v384-dp05-grad05-fg025-nbr" 7338 30 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " Part 3 (Jobs 17-24) successfully queued into SLURM!"
echo "=========================================================="
