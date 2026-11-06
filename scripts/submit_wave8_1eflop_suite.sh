#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_seg_boost.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting Wave 8: 1.0 EFLOP Nanobrain Synergy Suite"
echo " Target: Break Walnut (0.6480) on Official All-19 Benchmark"
echo " Strict Last-Layer Evaluation Protocol (No Intermediate Layers)"
echo " Guardrails: cpus-per-task=8, mem=48G, time=3:30:00"
echo " Full cluster access: nodes n-1 to n-8"
echo "=========================================================="

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

# ------------------------------------------------------------
# 1. w8-01-grad05-nbr: The Gradient Loss + Dense Neighbor Synergy
# Fuses our #1 1.0 EFLOP model (0.6473) with 3D dense neighbor context aggregation
# Budget: 15 epochs @ 384 = 1.0 EFLOP
# ------------------------------------------------------------
submit_train "w8-01-grad-nbr" "1.0e/w1024-d8-v384-grad05-nbr" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 2. w8-02-grad05-fg025-nbr: Gradient Boundary + Foreground Density Synergy
# Adds parenchymal contrast weighting (fg=0.25) to boost nerve/vessel Dice
# Budget: 15 epochs @ 384 = 1.0 EFLOP
# ------------------------------------------------------------
submit_train "w8-02-fg-nbr" "1.0e/w1024-d8-v384-grad05-fg025-nbr" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.fg_loss_weight=0.25 model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 3. w8-03-dec3-grad-nbr: Compact Decoder + Gradient Loss + Neighbor Pooling
# Dec=3 (dim=384) shallow decoder forces encoder representation density
# Budget: 16 epochs @ 384 = 1.0 EFLOP
# ------------------------------------------------------------
submit_train "w8-03-dec3-nbr" "1.0e/w1024-d8-v384-dec3-grad05-nbr" 7338 16 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=3 model_kwargs.decoder_embed_dim=384 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 4. w8-04-w1152-grad-nbr: 1152-Wide Architecture Frontier at 1.0 EFLOP
# Tests wider embedding capacity (1152 vs 1024) within strict 1.0 EFLOP budget
# Budget: 12 epochs @ 384 = 1.0 EFLOP
# ------------------------------------------------------------
submit_train "w8-04-w1152-nbr" "1.0e/w1152-d8-v384-grad05-nbr" 7338 12 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " All 4 Wave 8 (1.0 EFLOP) experiments successfully queued into SLURM!"
echo "=========================================================="

