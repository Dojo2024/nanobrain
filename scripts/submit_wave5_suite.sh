#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_EVAL="/admin/home/dojo/nanobrain/scripts/eval_seg_boost.sbatch"
S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_seg_boost.sbatch"

chmod +x "${S_EVAL}" "${S_TRAIN}"

echo "=========================================================="
echo " Submitting Wave 5: Frontier Optimization Suite (1.0 EFLOP)"
echo " Guardrails: cpus-per-task=8, mem=48G, time=3:30:00"
echo " NO NODE EXCLUSIONS: full cluster access (n-1 to n-8)"
echo "=========================================================="

# Nanobrain evaluation strictly on the last layer (no intermediate layers)
# EVAL_READOUT_LAYERS is deliberately omitted/disabled
export EVAL_NEIGHBOR_POOL="pyramid"
export EVAL_DECOUPLE_READOUT="true"
export EVAL_BRAIN_MASK_POOL="true"

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
# PART A: Native High-Resolution Frontiers (V=448 & V=512)
# Direct pretraining at dense patch resolution from epoch 0
# ------------------------------------------------------------
echo ""
echo "--- Part A: Submitting Native High-Resolution Pretraining ---"

# 1. w1024-d8-v448-grad05 (12 ep @ 448, lr=0.003, grad=0.5)
submit_train "w5-01-v448g50" "1.0e/w1024-d8-v448-grad05" 7338 12 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 2. w1024-d8-v448-lr35-grad05 (12 ep @ 448, lr=0.0035, grad=0.5) - Prime Contender
submit_train "w5-02-v448lr35" "1.0e/w1024-d8-v448-lr35-grad05" 7338 12 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 3. w1024-d8-v448-fg05-grad05 (12 ep @ 448, lr=0.0035, fg=0.5, grad=0.5)
submit_train "w5-03-v448fg05" "1.0e/w1024-d8-v448-fg05-grad05" 7338 12 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=0.5

# 4. w1024-d8-v448-fg10-grad05 (12 ep @ 448, lr=0.0035, fg=1.0, grad=0.5)
submit_train "w5-04-v448fg10" "1.0e/w1024-d8-v448-fg10-grad05" 7338 12 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=1.0

# 5. w1024-d8-v512-grad05 (10 ep @ 512, lr=0.003, grad=0.5)
submit_train "w5-05-v512g50" "1.0e/w1024-d8-v512-grad05" 7338 10 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=512 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 6. w1024-d8-v512-lr35-grad05 (10 ep @ 512, lr=0.0035, grad=0.5)
submit_train "w5-06-v512lr35" "1.0e/w1024-d8-v512-lr35-grad05" 7338 10 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=512 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# PART B: High-LR Calibrated V=384 Boundary Sweeps
# Combining winning V=384 resolution with optimized Muon LR (0.0035 - 0.0040)
# ------------------------------------------------------------
echo ""
echo "--- Part B: Submitting High-LR V=384 Pretraining ---"

# 7. w1024-d8-v384-lr35-grad05 (15 ep @ 384, lr=0.0035, grad=0.5)
submit_train "w5-07-v384lr35" "1.0e/w1024-d8-v384-lr35-grad05" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 8. w1024-d8-v384-lr40-grad05 (15 ep @ 384, lr=0.0040, grad=0.5)
submit_train "w5-08-v384lr40" "1.0e/w1024-d8-v384-lr40-grad05" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0040 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 9. w1024-d8-v384-lr35-fg05-grad05 (15 ep @ 384, lr=0.0035, fg=0.5, grad=0.5)
submit_train "w5-09-v384fg05" "1.0e/w1024-d8-v384-lr35-fg05-grad05" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=0.5

# 10. w1024-d8-v384-lr35-fg10-grad05 (15 ep @ 384, lr=0.0035, fg=1.0, grad=0.5)
submit_train "w5-10-v384fg10" "1.0e/w1024-d8-v384-lr35-fg10-grad05" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=1.0

# 11. w1024-d8-v384-warm3-lr35-grad05 (15 ep @ 384, warmup=3, lr=0.0035, grad=0.5)
submit_train "w5-11-v384w3lr35" "1.0e/w1024-d8-v384-warm3-lr35-grad05" 7338 15 3 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# PART C: Aspect Ratio Frontiers with High-LR
# ------------------------------------------------------------
echo ""
echo "--- Part C: Submitting Aspect Ratio Frontiers ---"

# 12. w1024-d9-v256-lr35-grad05 (16 ep @ 256, depth=9, width=1024, lr=0.0035, grad=0.5)
submit_train "w5-12-w1024d9lr35" "1.0e/w1024-d9-v256-lr35-grad05" 7338 16 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=9 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# 13. w1152-d7-v384-lr35-grad05 (14 ep @ 384, depth=7, width=1152, lr=0.0035, grad=0.5)
submit_train "w5-13-w1152d7lr35" "1.0e/w1152-d7-v384-lr35-grad05" 7338 14 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=7 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# PART D: Prohibited under new rule (Last layer only)
# ------------------------------------------------------------
echo "--- Part D: Omitted (Rule: evaluate only on last layer) ---"

echo "=========================================================="
echo " Wave 5 pretraining runs successfully queued into SLURM!"
echo "=========================================================="

