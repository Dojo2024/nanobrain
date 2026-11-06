#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_EVAL="/admin/home/dojo/nanobrain/scripts/eval_seg_boost.sbatch"
S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_seg_boost.sbatch"

chmod +x "${S_EVAL}" "${S_TRAIN}"

echo "=== Submitting Tier 1: Zero-Training Decoupled Multi-Scale Evals ==="

# E9: 1.0e/w1024-d8-v384-grad + Decoupled Readout (cat2468 + nbr3 for seg, layer 8 for global)
JOB_E9=$(sbatch -J segE9-decouple "${S_EVAL}" \
    "1.0e/w1024-d8-v384-grad" "1.0e/w1024-d8-v384-grad-decoupled" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" | awk '{print $NF}')
echo "Submitted E9 (grad-decoupled): Job ID ${JOB_E9}"

# E10: 1.0e/w1024-d8-v256 + Decoupled Readout
JOB_E10=$(sbatch -J segE10-v256dec "${S_EVAL}" \
    "1.0e/w1024-d8-v256" "1.0e/w1024-d8-v256-decoupled" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" | awk '{print $NF}')
echo "Submitted E10 (v256-decoupled): Job ID ${JOB_E10}"

# E11: 1.0e/w1024-d8-v384 base + Decoupled Readout
JOB_E11=$(sbatch -J segE11-basedec "${S_EVAL}" \
    "1.0e/w1024-d8-v384" "1.0e/w1024-d8-v384-decoupled" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" | awk '{print $NF}')
echo "Submitted E11 (base-decoupled): Job ID ${JOB_E11}"

echo ""
echo "=== Submitting Tier 2: 1.0-EFLOP Next-Gen Pretraining + Auto-Evals ==="

# P1: 1.0e w1024-d8-v256-grad (18 ep, v256 compression + gradient boundary loss)
JOB_P1=$(export EVAL_READOUT_LAYERS="[2,4,6,8]" EVAL_NEIGHBOR_POOL="true" EVAL_DECOUPLE_READOUT="true"; \
    sbatch -J segP1-v256-grad --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-grad" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P1 (v256-grad): Job ID ${JOB_P1}"

# P2: 1.0e w1024-d8-v320-grad (16 ep, balanced 86% masking ratio + gradient boundary loss)
JOB_P2=$(export EVAL_READOUT_LAYERS="[2,4,6,8]" EVAL_NEIGHBOR_POOL="true" EVAL_DECOUPLE_READOUT="true"; \
    sbatch -J segP2-v320-grad --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v320-grad" 7338 16 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=320 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P2 (v320-grad): Job ID ${JOB_P2}"

# P3: 1.0e w1024-d8-v384-fg-grad (15 ep, foreground-weighted MAE loss + gradient boundary loss)
JOB_P3=$(export EVAL_READOUT_LAYERS="[2,4,6,8]" EVAL_NEIGHBOR_POOL="true" EVAL_DECOUPLE_READOUT="true"; \
    sbatch -J segP3-fg-grad --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v384-fg-grad" 7338 15 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=1.5 | awk '{print $NF}')
echo "Submitted P3 (fg-grad): Job ID ${JOB_P3}"

echo ""
echo "All 6 jobs submitted safely with node protection (--exclude=n-3,n-4, cpus=8, mem=48G)."

