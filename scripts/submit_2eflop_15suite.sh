#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting Full 2.0 EFLOP Suite (15 Jobs with 16 CPUs/Task)"
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

# ============================================================
# GROUP A: The 3 Cancelled Prior Runs (Clean Restart)
# ============================================================

# 1. w2e-01-fg-nbr: The #1 Champion at 2.0 EFLOPs (30 ep)
submit_2e "w2e-01-fg-nbr" "2.0e/w1024-d8-v384-grad05-fg025-nbr" 7338 30 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 2. w2e-02-v256-nbr: High-Sample 92% Masking Compression (36 ep)
submit_2e "w2e-02-v256-nbr" "2.0e/w1024-d8-v256-grad05-fg025-nbr" 7338 36 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 3. w2e-03-w1152: Intermediate Width Frontier (24 ep)
submit_2e "w2e-03-w1152" "2.0e/w1152-d8-v384-grad05-nbr" 7338 24 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ============================================================
# GROUP B: The 4 Resubmitted Screenshot Runs
# ============================================================

# 4. w2e-05-w1024d12: Deep Capacity Scaling (22 ep)
submit_2e "w2e-05-w1024d12" "2.0e/w1024-d12-v384-grad05-nbr" 7338 22 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# 5. w2e-06-v512: Maximum Resolution Density (84% Masking) (20 ep)
submit_2e "w2e-06-v512" "2.0e/w1024-d8-v512-grad05-nbr" 7338 20 2 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=512 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# 6. w2e-07-curr: Dynamic Token Resolution Curriculum (V=256 -> V=448) (26 ep)
submit_2e "w2e-07-curr" "2.0e/curr-v256-to-v448-lr35-g50-nbr" 7338 26 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# 7. w2e-08-w1280d8: Ultra-Wide Capacity Frontier (18 ep)
submit_2e "w2e-08-w1280d8" "2.0e/w1280-d8-v384-grad05-nbr" 7338 18 2 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ============================================================
# GROUP C: 8 High-Impact Scaling Explorations at 2.0 EFLOPs
# ============================================================

# 8. w2e-09-fg-w1152: Width 1152 with Dual Boundary + Tissue Loss (24 ep)
submit_2e "w2e-09-fg-w1152" "2.0e/w1152-d8-v384-grad05-fg025-nbr" 7338 24 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 9. w2e-10-fg-d12: Depth 12 with Dual Boundary + Tissue Loss (22 ep)
submit_2e "w2e-10-fg-d12" "2.0e/w1024-d12-v384-grad05-fg025-nbr" 7338 22 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 10. w2e-11-grad075: Heavy Boundary Regularization (30 ep)
submit_2e "w2e-11-g75-nbr" "2.0e/w1024-d8-v384-grad075-nbr" 7338 30 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.75 \
    model_kwargs.dense_neighbor_pool=true

# 11. w2e-12-v320-fg-nbr: Balanced 90% Masking Sweet-Spot (32 ep)
submit_2e "w2e-12-v320-fg" "2.0e/w1024-d8-v320-grad05-fg025-nbr" 7338 32 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=320 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 12. w2e-13-v448dec3-fg: High Patch Density + Compact Decoder + Foreground Weighting (24 ep)
submit_2e "w2e-13-v448fg" "2.0e/w1024-d8-v448-dec3-g50-fg025-nbr" 7338 24 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=3 model_kwargs.decoder_embed_dim=384 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 13. w2e-14-v512-fg: Ultra-Fine Resolution + Dual Boundary/Tissue Loss (20 ep)
submit_2e "w2e-14-v512fg" "2.0e/w1024-d8-v512-grad05-fg025-nbr" 7338 20 2 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=512 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 14. w2e-15-w1152d10: Intermediate Wide-and-Deep Compromise (21 ep)
submit_2e "w2e-15-w1152d10" "2.0e/w1152-d10-v384-grad05-nbr" 7338 21 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# 15. w2e-16-w1280-fg: Ultra-Wide Capacity with Dual Boundary + Tissue Loss (18 ep)
submit_2e "w2e-16-w1280fg" "2.0e/w1280-d8-v384-grad05-fg025-nbr" 7338 18 2 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " All 15 2.0 EFLOP scaling jobs successfully queued into SLURM!"
echo "=========================================================="
