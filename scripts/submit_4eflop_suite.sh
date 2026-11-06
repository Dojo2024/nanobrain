#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting: 3 Resubmitted 2.0E Jobs + 16 Frontier 4.0E Jobs"
echo " Strict Last-Layer Evaluation Protocol (No Intermediate Layers)"
echo " Cluster Guardrails: cpus-per-task=16, mem=48G, requeue enabled"
echo " Full cluster access: nodes n-1 to n-8"
echo "=========================================================="

submit_job() {
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
# PART A: Resubmitted 2.0 EFLOP Jobs (3 Runs)
# ============================================================
echo "--- Part A: Submitting 3 Resubmitted 2.0 EFLOP Jobs ---"

# 1. w2e-44-d10curr: Deep Stack (D=10) with Resolution Curriculum (V=256 -> 384) (24 ep)
submit_job "w2e-44-d10curr" "2.0e/curr-d10-v256-to-v384-dp05-g50-fg025-nbr" 7338 24 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    cooldown_epochs=6 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 2. w2e-45-v320dp: Sweet-Spot Patch Resolution (V=320) + Stochastic Depth (28 ep)
submit_job "w2e-45-v320dp" "2.0e/w1024-d8-v320-dp05-grad05-fg025-nbr" 7338 28 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=320 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 3. w2e-46-d12lr35: Deep Stack (D=12) with Accelerated Warmup (lr=0.0035, warmup=4) (22 ep)
submit_job "w2e-46-d12lr35" "2.0e/w1024-d12-v384-lr35-warmup4-dp05-g50-nbr" 7338 22 4 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ============================================================
# PART B: 16 Frontier 4.0 EFLOP Jobs
# ============================================================
echo "--- Part B: Submitting 16 Frontier 4.0 EFLOP Jobs ---"

# 1. w4e-01-d8dp05: SOTA Champion at 4.0 EFLOP Envelope (60 ep)
submit_job "w4e-01-d8dp05" "4.0e/w1024-d8-v384-dp05-grad05-fg025-nbr" 7338 60 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 2. w4e-02-d12dp05: Deep Hierarchy (D=12) at 4.0 EFLOP Envelope (44 ep)
submit_job "w4e-02-d12dp05" "4.0e/w1024-d12-v384-dp05-grad05-fg025-nbr" 7338 44 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 3. w4e-03-d10dp05: Tumor-Core Record Holder (D=10) at 4.0 EFLOP Envelope (52 ep)
submit_job "w4e-03-d10dp05" "4.0e/w1024-d10-v384-dp05-grad05-fg025-nbr" 7338 52 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 4. w4e-04-d14dp04: Ultra-Deep Hierarchy (D=14) Scaling (38 ep)
submit_job "w4e-04-d14dp04" "4.0e/w1024-d14-v384-dp04-grad05-fg025-nbr" 7338 38 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0028 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=14 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.04 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 5. w4e-05-d16dp04: Frontier Depth Scaling (D=16) (34 ep)
submit_job "w4e-05-d16dp04" "4.0e/w1024-d16-v384-dp04-grad05-fg025-nbr" 7338 34 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0028 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=16 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.04 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 6. w4e-06-curr256-384: Extended Curriculum (V=256 -> V=384) (60 ep: 46 ep @ 256 + 14 ep @ 384)
submit_job "w4e-06-curr256" "4.0e/curr-v256-to-v384-dp05-g50-fg025-nbr" 7338 60 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    cooldown_epochs=14 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 7. w4e-07-curr256-448: High-Density Curriculum (V=256 -> V=448) (58 ep: 46 ep @ 256 + 12 ep @ 448)
submit_job "w4e-07-curr448" "4.0e/curr-v256-to-v448-dp05-g50-fg025-nbr" 7338 58 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    cooldown_epochs=12 cooldown_num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 8. w4e-08-d10curr: Hierarchical Curriculum (D=10, V=256 -> V=384) (50 ep: 38 ep @ 256 + 12 ep @ 384)
submit_job "w4e-08-d10curr" "4.0e/curr-d10-v256-to-v384-dp05-g50-fg025-nbr" 7338 50 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    cooldown_epochs=12 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 9. w4e-09-w1152d10: Dual Width + Depth Scaling (W=1152, D=10) (42 ep)
submit_job "w4e-09-w1152d10" "4.0e/w1152-d10-v384-dp05-grad05-fg025-nbr" 7338 42 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 10. w4e-10-w1152d12: Deep Wide Capacity (W=1152, D=12) (36 ep)
submit_job "w4e-10-w1152d12" "4.0e/w1152-d12-v384-dp05-grad05-fg025-nbr" 7338 36 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0028 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 11. w4e-11-w1280d8: Massive Width Channel Capacity (W=1280, D=8) (36 ep)
submit_job "w4e-11-w1280d8" "4.0e/w1280-d8-v384-dp06-grad05-fg025-nbr" 7338 36 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0028 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.06 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 12. w4e-12-w1280d10: Frontier High-Capacity Network (W=1280, D=10) (30 ep)
submit_job "w4e-12-w1280d10" "4.0e/w1280-d10-v384-dp06-grad05-fg025-nbr" 7338 30 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0028 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.06 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 13. w4e-13-v256dp05: Extended Phenotypic Age Horizon (V=256) (72 ep)
submit_job "w4e-13-v256dp05" "4.0e/w1024-d8-v256-dp05-grad05-fg025-nbr" 7338 72 5 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 14. w4e-14-v320dp05: Sweet-Spot Resolution Scaling (V=320) (64 ep)
submit_job "w4e-14-v320dp05" "4.0e/w1024-d8-v320-dp05-grad05-fg025-nbr" 7338 64 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=320 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# 15. w4e-15-d8g60fg30: Enhanced Boundary & Foreground Regularization (60 ep)
submit_job "w4e-15-d8g60fg30" "4.0e/w1024-d8-v384-dp06-grad060-fg030-nbr" 7338 60 4 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.06 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.60 \
    model_kwargs.fg_loss_weight=0.30 model_kwargs.dense_neighbor_pool=true

# 16. w4e-16-d12lr35: Deep Stack Accelerated Convergence (D=12, lr=0.0035, warmup=5) (44 ep)
submit_job "w4e-16-d12lr35" "4.0e/w1024-d12-v384-lr35-warmup5-dp05-g50-nbr" 7338 44 5 "5.2h" "07:30:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " All 19 scaling jobs (3 resubmissions + 16 new 4.0E) successfully queued!"
echo "=========================================================="
