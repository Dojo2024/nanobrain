#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting Part 6: 8 Frontier 2.0 EFLOP Scaling Jobs"
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
# 1. w2e-39-curr256-384dp: Resolution Curriculum (V=256 -> V=384) + Stochastic Depth (26 ep)
# Fuses our #1 Phenotypic Age model (V=256, Record R2=0.8994) with #1 Seg model (V=384, Core-5=0.7059)
# ------------------------------------------------------------
submit_2e "w2e-39-curr256dp" "2.0e/curr-v256-to-v384-dp05-g50-fg025-nbr" 7338 26 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    cooldown_epochs=6 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 2. w2e-40-d10v256dp: Deeper Hierarchy (D=10) with Compact Patch Resolution (V=256) (26 ep)
# Deepens the representation stack to test if D=10 pushes Age-2 R2 beyond 0.900
# ------------------------------------------------------------
submit_2e "w2e-40-d10v256dp" "2.0e/w1024-d10-v256-dp05-grad05-fg025-nbr" 7338 26 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 3. w2e-41-d12dp03fg: Deep Hierarchy (D=12) with Gentle Stochastic Depth (dp=0.03) (22 ep)
# Avoids over-regularizing the 12-layer deep stack while retaining co-adaptation immunity
# ------------------------------------------------------------
submit_2e "w2e-41-d12dp03fg" "2.0e/w1024-d12-v384-dp03-grad05-fg025-nbr" 7338 22 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.03 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 4. w2e-42-w1152d10dp: Dual Capacity Expansion (W=1152, D=10) + Drop Path (22 ep)
# Simultaneously expands channel capacity (+25%) and hierarchical depth
# ------------------------------------------------------------
submit_2e "w2e-42-w1152d10" "2.0e/w1152-d10-v384-dp05-grad05-fg025-nbr" 7338 22 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 5. w2e-43-d8g60fg30: SOTA Champion Fine-Tuning (grad=0.60, fg=0.30) (30 ep)
# Tests boosted edge boundary and foreground focal loss on our all-time SOTA backbone
# ------------------------------------------------------------
submit_2e "w2e-43-d8g60fg30" "2.0e/w1024-d8-v384-dp05-grad060-fg030-nbr" 7338 30 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.60 \
    model_kwargs.fg_loss_weight=0.30 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 6. w2e-44-d10curr: Deep Stack (D=10) with Resolution Curriculum (V=256 -> 384) (24 ep)
# ------------------------------------------------------------
submit_2e "w2e-44-d10curr" "2.0e/curr-d10-v256-to-v384-dp05-g50-fg025-nbr" 7338 24 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=256 \
    cooldown_epochs=6 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=10 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 7. w2e-45-v320dp: Sweet-Spot Patch Resolution (V=320) + Stochastic Depth (28 ep)
# Optimal balance between global volume perception and localized tumor/stroke detail
# ------------------------------------------------------------
submit_2e "w2e-45-v320dp" "2.0e/w1024-d8-v320-dp05-grad05-fg025-nbr" 7338 28 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=320 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 8. w2e-46-d12lr35: Deep Stack (D=12) with Accelerated Warmup (lr=0.0035, warmup=4) (22 ep)
# Accelerates representation formation in deeper 12-layer stack under Muon optimizer
# ------------------------------------------------------------
submit_2e "w2e-46-d12lr35" "2.0e/w1024-d12-v384-lr35-warmup4-dp05-g50-nbr" 7338 22 4 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " All 8 Part 6 scaling jobs successfully queued into SLURM!"
echo "=========================================================="
