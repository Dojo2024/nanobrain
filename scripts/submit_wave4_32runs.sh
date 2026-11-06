#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_EVAL="/admin/home/dojo/nanobrain/scripts/eval_seg_boost.sbatch"
S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_seg_boost.sbatch"

chmod +x "${S_EVAL}" "${S_TRAIN}"

echo "=========================================================="
echo " Submitting Wave 4: 32-Job 1.0 EFLOP Optimization Suite"
echo " Strict Guardrails: cpus-per-task=8, mem=48G, time=3:30:00"
echo " NO NODE EXCLUSIONS: full cluster access (n-1 to n-8)"
echo "=========================================================="

# Standard Grand Slam readout environment for all evaluations
export EVAL_READOUT_LAYERS="[1,2,4,6,8]"
export EVAL_NEIGHBOR_POOL="pyramid"
export EVAL_DECOUPLE_READOUT="true"
export EVAL_BRAIN_MASK_POOL="true"

# Helper for submitting training jobs
submit_train() {
    local job_name="$1"
    local run_name="$2"
    local seed="$3"
    local epochs="$4"
    local warmup="$5"
    local time_limit="$6"
    shift 6
    local job_id
    job_id=$(sbatch -J "${job_name}" --time=3:30:00 "${S_TRAIN}" "${run_name}" "${seed}" "${epochs}" "${warmup}" "${time_limit}" "$@" | awk '{print $NF}')
    echo "Submitted ${job_name} (${run_name}): Job ID ${job_id}"
}

# Helper for submitting evaluation catch-up jobs
submit_eval() {
    local job_name="$1"
    local src_run="$2"
    local dest_run="$3"
    local seed="$4"
    shift 4
    local job_id
    job_id=$(sbatch -J "${job_name}" --time=1:00:00 "${S_EVAL}" "${src_run}" "${dest_run}" "${seed}" "$@" | awk '{print $NF}')
    echo "Submitted ${job_name} (${dest_run}): Job ID ${job_id}"
}

# ------------------------------------------------------------
# CLUSTER 1: Resolution Curriculum Pretraining (6 jobs)
# Combines V=224/256 global semantics with V=384/448/512 edge sharpness
# ------------------------------------------------------------
echo ""
echo "--- Submitting Cluster 1: Resolution Curriculum (Jobs 1-6) ---"

# 1. curr-v224-to-v384-grad05 (14 ep @ 224 + 3 ep @ 384, grad=0.5)
submit_train "w4-01-c224to384" "1.0e/curr-v224-to-v384-grad05" 7338 17 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=224 \
    cooldown_epochs=3 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 2. curr-v224-to-v384-fg10-grad05 (14 ep @ 224 + 3 ep @ 384, fg=1.0, grad=0.5)
submit_train "w4-02-c224to384fg" "1.0e/curr-v224-to-v384-fg10-grad05" 7338 17 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=224 \
    cooldown_epochs=3 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=1.0

# 3. curr-v256-to-v384-grad05 (13 ep @ 256 + 3 ep @ 384, grad=0.5)
submit_train "w4-03-c256to384" "1.0e/curr-v256-to-v384-grad05" 7338 16 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    cooldown_epochs=3 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 4. curr-v256-to-v448-grad05 (14 ep @ 256 + 2 ep @ 448, grad=0.5)
submit_train "w4-04-c256to448" "1.0e/curr-v256-to-v448-grad05" 7338 16 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    cooldown_epochs=2 cooldown_num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 5. curr-v224-to-v448-grad05 (15 ep @ 224 + 2 ep @ 448, grad=0.5)
submit_train "w4-05-c224to448" "1.0e/curr-v224-to-v448-grad05" 7338 17 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=224 \
    cooldown_epochs=2 cooldown_num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 6. curr-v256-to-v512-grad05 (14 ep @ 256 + 2 ep @ 512, grad=0.5)
submit_train "w4-06-c256to512" "1.0e/curr-v256-to-v512-grad05" 7338 16 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    cooldown_epochs=2 cooldown_num_visible=512 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# CLUSTER 2: V=384 Dense Boundary Refinements (6 jobs)
# Pushing Core-5 segmentation directly beyond Walnut's 0.7220
# ------------------------------------------------------------
echo ""
echo "--- Submitting Cluster 2: V=384 Boundary Refinements (Jobs 7-12) ---"

# 7. v384-grad035 (15 ep @ 384, grad=0.35)
submit_train "w4-07-v384g35" "1.0e/w1024-d8-v384-grad035" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.35

# 8. v384-grad065 (15 ep @ 384, grad=0.65)
submit_train "w4-08-v384g65" "1.0e/w1024-d8-v384-grad065" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.65

# 9. v384-fg05-grad05 (15 ep @ 384, fg=0.5, grad=0.5)
submit_train "w4-09-v384fg05" "1.0e/w1024-d8-v384-fg05-grad05" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=0.5

# 10. v384-fg10-grad05 (15 ep @ 384, fg=1.0, grad=0.5)
submit_train "w4-10-v384fg10" "1.0e/w1024-d8-v384-fg10-grad05" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=1.0

# 11. v384-fg15-grad05 (15 ep @ 384, fg=1.5, grad=0.5)
submit_train "w4-11-v384fg15" "1.0e/w1024-d8-v384-fg15-grad05" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=1.5

# 12. v384-warmup3-grad05 (15 ep @ 384, warmup=3, grad=0.5)
submit_train "w4-12-v384warm3" "1.0e/w1024-d8-v384-warmup3-grad05" 7338 15 3 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# CLUSTER 3: V=224 High-Masking Global Semantics (6 jobs)
# Deep global representation learning with 90.3% masking
# ------------------------------------------------------------
echo ""
echo "--- Submitting Cluster 3: V=224 High-Masking Semantics (Jobs 13-18) ---"

# 13. v224-grad035 (20 ep @ 224, grad=0.35)
submit_train "w4-13-v224g35" "1.0e/w1024-d8-v224-grad035" 7338 20 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=224 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.35

# 14. v224-grad025 (20 ep @ 224, grad=0.25)
submit_train "w4-14-v224g25" "1.0e/w1024-d8-v224-grad025" 7338 20 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=224 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.25

# 15. v224-fg05-grad05 (20 ep @ 224, fg=0.5, grad=0.5)
submit_train "w4-15-v224fg05" "1.0e/w1024-d8-v224-fg05-grad05" 7338 20 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=224 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=0.5

# 16. v224-fg10-grad05 (20 ep @ 224, fg=1.0, grad=0.5)
submit_train "w4-16-v224fg10" "1.0e/w1024-d8-v224-fg10-grad05" 7338 20 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=224 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=1.0

# 17. v224-warmup3-grad05 (20 ep @ 224, warmup=3, grad=0.5)
submit_train "w4-17-v224warm3" "1.0e/w1024-d8-v224-warmup3-grad05" 7338 20 3 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=224 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 18. v224-lr35e3-grad05 (20 ep @ 224, base_lr=0.0035, grad=0.5)
submit_train "w4-18-v224lr35" "1.0e/w1024-d8-v224-lr35e3-grad05" 7338 20 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=224 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# CLUSTER 4: Aspect Ratios & Capacity Allocation (6 jobs)
# Testing width vs depth trade-offs
# ------------------------------------------------------------
echo ""
echo "--- Submitting Cluster 4: Aspect Ratios (Jobs 19-24) ---"

# 19. w1024-d9-v256-grad05 (16 ep @ 256, depth=9, width=1024, grad=0.5)
submit_train "w4-19-w1024d9" "1.0e/w1024-d9-v256-grad05" 7338 16 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=9 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 20. w1152-d8-v256-grad05 (16 ep @ 256, depth=8, width=1152, grad=0.5)
submit_train "w4-20-w1152d8" "1.0e/w1152-d8-v256-grad05" 7338 16 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 21. w896-d8-v384-grad05 (17 ep @ 384, depth=8, width=896, grad=0.5)
submit_train "w4-21-w896d8" "1.0e/w896-d8-v384-grad05" 7338 17 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=896 model_kwargs.depth=8 model_kwargs.num_heads=14 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 22. w896-d9-v320-grad05 (17 ep @ 320, depth=9, width=896, grad=0.5)
submit_train "w4-22-w896d9" "1.0e/w896-d9-v320-grad05" 7338 17 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=320 \
    model_kwargs.embed_dim=896 model_kwargs.depth=9 model_kwargs.num_heads=14 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 23. w960-d8-v384-grad05 (16 ep @ 384, depth=8, width=960, grad=0.5)
submit_train "w4-23-w960d8" "1.0e/w960-d8-v384-grad05" 7338 16 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=960 model_kwargs.depth=8 model_kwargs.num_heads=15 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 24. w1152-d7-v384-grad05 (14 ep @ 384, depth=7, width=1152, grad=0.5)
submit_train "w4-24-w1152d7" "1.0e/w1152-d7-v384-grad05" 7338 14 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=7 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# CLUSTER 5: Optimization Dynamics & Regularization (4 jobs)
# LR scaling and stochastic depth decay
# ------------------------------------------------------------
echo ""
echo "--- Submitting Cluster 5: Optimization & Regularization (Jobs 25-28) ---"

# 25. v384-muon-lr35e3-warmup2 (15 ep @ 384, base_lr=0.0035, warmup=2, grad=0.5)
submit_train "w4-25-v384lr35" "1.0e/w1024-d8-v384-muon-lr35e3-warmup2" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 26. v224-muon-lr4e3-warmup2 (20 ep @ 224, base_lr=0.0040, warmup=2, grad=0.5)
submit_train "w4-26-v224lr40" "1.0e/w1024-d8-v224-muon-lr4e3-warmup2" 7338 20 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0040 num_visible=224 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 27. v384-droppath005-grad05 (15 ep @ 384, drop_path_rate=0.05, grad=0.5)
submit_train "w4-27-v384dp05" "1.0e/w1024-d8-v384-droppath005-grad05" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 28. v256-droppath005-grad05 (18 ep @ 256, drop_path_rate=0.05, grad=0.5)
submit_train "w4-28-v256dp05" "1.0e/w1024-d8-v256-droppath005-grad05" 7338 18 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.drop_path_rate=0.05 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# CLUSTER 6: Decoder Capacity & Frontier Catch-Up (4 jobs)
# Heavier decoders + completing the 4 partially evaluated runs
# ------------------------------------------------------------
echo ""
echo "--- Submitting Cluster 6: Decoders & Frontier Catch-Up (Jobs 29-32) ---"

# 29. v384-dec3-w512-grad05 (14 ep @ 384, decoder_depth=3, decoder_embed_dim=512)
submit_train "w4-29-v384dec3" "1.0e/w1024-d8-v384-dec3-w512-grad05" 7338 14 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=3 model_kwargs.decoder_embed_dim=512 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 30. v224-dec3-w512-grad05 (19 ep @ 224, decoder_depth=3, decoder_embed_dim=512)
submit_train "w4-30-v224dec3" "1.0e/w1024-d8-v224-dec3-w512-grad05" 7338 19 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=224 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=3 model_kwargs.decoder_embed_dim=512 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 31. fix-v256cool384: complete remaining 12 tasks on w1024-d8-v256-cool-v384
submit_eval "w4-31-fix-cool384" "1.0e/w1024-d8-v256-cool-v384" "1.0e/w1024-d8-v256-cool-v384" 7338 \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 32. fix-v256cool512: complete remaining 6 tasks on w1024-d8-v256-cool-v512
submit_eval "w4-32-fix-cool512" "1.0e/w1024-d8-v256-cool-v512" "1.0e/w1024-d8-v256-cool-v512" 7338 \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

echo "=========================================================="
echo " All 32 runs successfully queued into SLURM!"
echo "=========================================================="

