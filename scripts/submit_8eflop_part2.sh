#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting Part 2: 8 More Frontier 8.0 EFLOP Scaling Jobs"
echo " Incorporating Breakthrough 4.0E Discoveries (D14/D16, W1280-D10, D10-Curr)"
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
# 9. w8e-09-d10curr: The All-Time #1 SOTA Champion (0.6583 All-19 at 4.0E)
# D=10 Hierarchy + Resolution Curriculum (V=256 -> 384) (100 ep: 76 ep @ 256 + 24 ep @ 384 cooldown)
# ------------------------------------------------------------
submit_8e "w8e-09-d10curr" "8.0e/curr-d10-v256-to-v384-dp05-g50-fg025-nbr" 7338 100 7 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    cooldown_epochs=24 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 10. w8e-10-d14dp04: The #2 SOTA Champion & Deep Frontier (0.6575 All-19, 0.7125 Core-5 at 4.0E)
# D=14 Deep Hierarchy + dp=0.04 (76 ep)
# ------------------------------------------------------------
submit_8e "w8e-10-d14dp04" "8.0e/w1024-d14-v384-dp04-grad05-fg025-nbr" 7338 76 6 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0028 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=14 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.04 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 11. w8e-11-d16dp04: Ultra-Deep Hierarchy Frontier (0.6573 All-19, 0.7104 Core-5 at 4.0E)
# D=16 Ultra-Deep Hierarchy + dp=0.04 (68 ep)
# ------------------------------------------------------------
submit_8e "w8e-11-d16dp04" "8.0e/w1024-d16-v384-dp04-grad05-fg025-nbr" 7338 68 6 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0028 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=16 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.04 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 12. w8e-12-w1280d10: Historic Segmentation Champion (0.7158 Core-5, 0.8096 TC Dice at 4.0E)
# W=1280 Ultra-Wide + D=10 Hierarchy + dp=0.06 (76 ep)
# ------------------------------------------------------------
submit_8e "w8e-12-w1280d10" "8.0e/w1280-d10-v384-dp06-grad05-fg025-nbr" 7338 76 6 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0028 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.06 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 13. w8e-13-w1152d12: Wide + Deep Sweet Spot (0.6566 All-19, 0.7091 Core-5, 0.8953 Age-2 at 4.0E)
# W=1152 + D=12 Stack + dp=0.05 (76 ep)
# ------------------------------------------------------------
submit_8e "w8e-13-w1152d12" "8.0e/w1152-d12-v384-dp05-grad05-fg025-nbr" 7338 76 6 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0028 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 14. w8e-14-w1152d10: Wide + Intermediate Depth Hierarchy (0.6534 All-19, 0.7114 Core-5 at 4.0E)
# W=1152 + D=10 Stack + dp=0.05 (84 ep)
# ------------------------------------------------------------
submit_8e "w8e-14-w1152d10" "8.0e/w1152-d10-v384-dp05-grad05-fg025-nbr" 7338 84 6 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 15. w8e-15-w1280d8: Wide Channel + Regularized Backbone (0.6560 All-19, 0.7080 Core-5 at 4.0E)
# W=1280 + D=8 Stack + dp=0.06 (88 ep)
# ------------------------------------------------------------
submit_8e "w8e-15-w1280d8" "8.0e/w1280-d8-v384-dp06-grad05-fg025-nbr" 7338 88 6 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.06 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 16. w8e-16-d8g60fg30: Long-Horizon Boundary Tuning (grad=0.60, fg=0.30, dp=0.06) (120 ep)
# D=8, V=384 with heavy boundary loss & stochastic depth tuned for 120 ep
# ------------------------------------------------------------
submit_8e "w8e-16-d8g60fg30" "8.0e/w1024-d8-v384-dp06-grad060-fg030-nbr" 7338 120 8 "${TIME_LIMIT}" "${SLURM_TIME}" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.06 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.60 \
    model_kwargs.fg_loss_weight=0.30 model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " All 8 Frontier Part 2 8.0 EFLOP jobs successfully queued into SLURM!"
echo "=========================================================="
