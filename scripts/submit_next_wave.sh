#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_EVAL="/admin/home/dojo/nanobrain/scripts/eval_seg_boost.sbatch"
S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_seg_boost.sbatch"

chmod +x "${S_EVAL}" "${S_TRAIN}"

echo "=== Submitting Wave 2: Readout Innovations (Runs R01-R07) ==="

# R01: 1.0e/w1024-d8-v256 + brain-masked foreground global pooling
JOB_R01=$(sbatch -J segR01-v256fg "${S_EVAL}" \
    "1.0e/w1024-d8-v256" "1.0e/w1024-d8-v256-fgpool" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true" | awk '{print $NF}')
echo "Submitted R01 (v256-fgpool): Job ID ${JOB_R01}"

# R02: 1.0e/w1024-d8-v384-grad + 5-layer dense readout [1,2,4,6,8]
JOB_R02=$(sbatch -J segR02-cat12468 "${S_EVAL}" \
    "1.0e/w1024-d8-v384-grad" "1.0e/w1024-d8-v384-grad-cat12468" 7338 \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" | awk '{print $NF}')
echo "Submitted R02 (grad-cat12468): Job ID ${JOB_R02}"

# R03: 1.0e/w1024-d8-v384-grad + staggered dense readout [3,5,7,8]
JOB_R03=$(sbatch -J segR03-cat3578 "${S_EVAL}" \
    "1.0e/w1024-d8-v384-grad" "1.0e/w1024-d8-v384-grad-cat3578" 7338 \
    "model_kwargs.readout_layers=[3,5,7,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" | awk '{print $NF}')
echo "Submitted R03 (grad-cat3578): Job ID ${JOB_R03}"

# R04: 1.0e/w1024-d8-v384-grad + 3D Spatial Pyramid Pooling (pyramid nbr)
JOB_R04=$(sbatch -J segR04-spp "${S_EVAL}" \
    "1.0e/w1024-d8-v384-grad" "1.0e/w1024-d8-v384-grad-spp" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" | awk '{print $NF}')
echo "Submitted R04 (grad-spp): Job ID ${JOB_R04}"

# R06: 1.0e/w1024-d8-v256 + dual-layer global readout [7+8]
JOB_R06=$(sbatch -J segR06-dual "${S_EVAL}" \
    "1.0e/w1024-d8-v256" "1.0e/w1024-d8-v256-dual" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.dual_global_readout=true" | awk '{print $NF}')
echo "Submitted R06 (v256-dual): Job ID ${JOB_R06}"

# R07: 1.0e/w1024-d8-v384-grad + SPP + Foreground-Masked Pool
JOB_R07=$(sbatch -J segR07-fgspp "${S_EVAL}" \
    "1.0e/w1024-d8-v384-grad" "1.0e/w1024-d8-v384-grad-fgpool-spp" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true" | awk '{print $NF}')
echo "Submitted R07 (grad-fgpool-spp): Job ID ${JOB_R07}"

echo ""
echo "=== Submitting Wave 2: Next Pretraining Innovations (Runs P4-P7) ==="

# P4: 1.0e w1024-d8-v256 + fg_loss_weight=2.0 + grad_loss_weight=0.5 (18 ep)
JOB_P4=$(export EVAL_READOUT_LAYERS="[2,4,6,8]" EVAL_NEIGHBOR_POOL="true" EVAL_DECOUPLE_READOUT="true"; \
    sbatch -J segP4-fg20 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-fg20" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=2.0 | awk '{print $NF}')
echo "Submitted P4 (v256-fg20): Job ID ${JOB_P4}"

# P5: 1.0e w1024-d8-v256 + fg_loss_weight=1.5 + grad_loss_weight=0.35 (18 ep)
JOB_P5=$(export EVAL_READOUT_LAYERS="[2,4,6,8]" EVAL_NEIGHBOR_POOL="true" EVAL_DECOUPLE_READOUT="true"; \
    sbatch -J segP5-fg15-g35 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-fg15-grad035" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.35 model_kwargs.fg_loss_weight=1.5 | awk '{print $NF}')
echo "Submitted P5 (v256-fg15-grad035): Job ID ${JOB_P5}"

# P6: 1.0e w1024-d8-v224-grad (20 ep, 90% masking ratio)
JOB_P6=$(export EVAL_READOUT_LAYERS="[2,4,6,8]" EVAL_NEIGHBOR_POOL="true" EVAL_DECOUPLE_READOUT="true"; \
    sbatch -J segP6-v224-grad --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v224-grad" 7338 20 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=224 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P6 (v224-grad): Job ID ${JOB_P6}"

# P7: 1.0e w1024-d8-v288-grad (17 ep, 87% masking ratio)
JOB_P7=$(export EVAL_READOUT_LAYERS="[2,4,6,8]" EVAL_NEIGHBOR_POOL="true" EVAL_DECOUPLE_READOUT="true"; \
    sbatch -J segP7-v288-grad --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v288-grad" 7338 17 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=288 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P7 (v288-grad): Job ID ${JOB_P7}"

echo ""
echo "Wave 2 submitted safely with node guards (--exclude=n-3,n-4, cpus=8, mem=48G)."

