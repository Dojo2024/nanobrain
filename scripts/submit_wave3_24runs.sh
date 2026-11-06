#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_EVAL="/admin/home/dojo/nanobrain/scripts/eval_seg_boost.sbatch"
S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_seg_boost.sbatch"

chmod +x "${S_EVAL}" "${S_TRAIN}"

echo "=========================================================="
echo " Submitting Master 24-Run IsoFLOP Optimization Suite (1.0 EFLOP)"
echo " Guardrails: cpus-per-task=8, mem=48G"
echo "=========================================================="

# ------------------------------------------------------------
# PART A: Readout Synergy Evaluations on Champion Checkpoints (10 runs)
# ------------------------------------------------------------
echo "--- Part A: Submitting 10 Targeted Readout Evaluations ---"

# 1. v256-grad + 5-layer [1,2,4,6,8] + SPP pyramid
JOB_E01=$(sbatch -J w3E01-v256-cat12468-spp "${S_EVAL}" \
    "1.0e/w1024-d8-v256-grad" "1.0e/w1024-d8-v256-grad-cat12468-spp" 7338 \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" | awk '{print $NF}')
echo "Submitted E01 (v256-grad-cat12468-spp): Job ID ${JOB_E01}"

# 2. v256-grad + brain mask global pool
JOB_E02=$(sbatch -J w3E02-v256-fgpool "${S_EVAL}" \
    "1.0e/w1024-d8-v256-grad" "1.0e/w1024-d8-v256-grad-fgpool" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true" | awk '{print $NF}')
echo "Submitted E02 (v256-grad-fgpool): Job ID ${JOB_E02}"

# 3. v256-grad + 5-layer [1,2,4,6,8] + SPP pyramid + brain mask global pool (Grand Slam)
JOB_E03=$(sbatch -J w3E03-v256-slam "${S_EVAL}" \
    "1.0e/w1024-d8-v256-grad" "1.0e/w1024-d8-v256-grad-slam" 7338 \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true" | awk '{print $NF}')
echo "Submitted E03 (v256-grad-slam): Job ID ${JOB_E03}"

# 4. v256-grad + [2,4,6,8] + SPP pyramid
JOB_E04=$(sbatch -J w3E04-v256-spp "${S_EVAL}" \
    "1.0e/w1024-d8-v256-grad" "1.0e/w1024-d8-v256-grad-spp" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" | awk '{print $NF}')
echo "Submitted E04 (v256-grad-spp): Job ID ${JOB_E04}"

# 5. v320-grad + 5-layer [1,2,4,6,8] + SPP pyramid
JOB_E05=$(sbatch -J w3E05-v320-cat12468-spp "${S_EVAL}" \
    "1.0e/w1024-d8-v320-grad" "1.0e/w1024-d8-v320-grad-cat12468-spp" 7338 \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" | awk '{print $NF}')
echo "Submitted E05 (v320-grad-cat12468-spp): Job ID ${JOB_E05}"

# 6. v320-grad + brain mask global pool
JOB_E06=$(sbatch -J w3E06-v320-fgpool "${S_EVAL}" \
    "1.0e/w1024-d8-v320-grad" "1.0e/w1024-d8-v320-grad-fgpool" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true" | awk '{print $NF}')
echo "Submitted E06 (v320-grad-fgpool): Job ID ${JOB_E06}"

# 7. v320-grad + 5-layer [1,2,4,6,8] + SPP pyramid + brain mask global pool
JOB_E07=$(sbatch -J w3E07-v320-slam "${S_EVAL}" \
    "1.0e/w1024-d8-v320-grad" "1.0e/w1024-d8-v320-grad-slam" 7338 \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true" | awk '{print $NF}')
echo "Submitted E07 (v320-grad-slam): Job ID ${JOB_E07}"

# 8. v384-fg-grad + 5-layer [1,2,4,6,8] + SPP pyramid
JOB_E08=$(sbatch -J w3E08-v384fg-cat12468-spp "${S_EVAL}" \
    "1.0e/w1024-d8-v384-fg-grad" "1.0e/w1024-d8-v384-fg-grad-cat12468-spp" 7338 \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" | awk '{print $NF}')
echo "Submitted E08 (v384fg-cat12468-spp): Job ID ${JOB_E08}"

# 9. v384-fg-grad + brain mask global pool
JOB_E09=$(sbatch -J w3E09-v384fg-fgpool "${S_EVAL}" \
    "1.0e/w1024-d8-v384-fg-grad" "1.0e/w1024-d8-v384-fg-grad-fgpool" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true" | awk '{print $NF}')
echo "Submitted E09 (v384fg-fgpool): Job ID ${JOB_E09}"

# 10. v384-grad + 5-layer [1,2,4,6,8] + brain mask global pool
JOB_E10=$(sbatch -J w3E10-v384-cat12468-fgpool "${S_EVAL}" \
    "1.0e/w1024-d8-v384-grad" "1.0e/w1024-d8-v384-grad-cat12468-fgpool" 7338 \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true" | awk '{print $NF}')
echo "Submitted E10 (v384-cat12468-fgpool): Job ID ${JOB_E10}"

echo ""
# ------------------------------------------------------------
# PART B: 14 IsoFLOP Pretraining Runs (Strictly <= 1.0 EFLOP)
# ------------------------------------------------------------
echo "--- Part B: Submitting 14 Pretraining Runs with Grand Slam Readout ---"

# Standard readout environment for post-pretrain evaluations
export EVAL_READOUT_LAYERS="[1,2,4,6,8]"
export EVAL_NEIGHBOR_POOL="pyramid"
export EVAL_DECOUPLE_READOUT="true"
export EVAL_BRAIN_MASK_POOL="true"

# 11. w1024-d8-v256-fg15-grad05 (18 ep, v256, fg=1.5, grad=0.5)
JOB_P08=$(sbatch -J w3P08-fg15g50 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-fg15-grad05" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=1.5 | awk '{print $NF}')
echo "Submitted P08 (v256-fg15-grad05): Job ID ${JOB_P08}"

# 12. w1024-d8-v256-fg10-grad05 (18 ep, v256, fg=1.0, grad=0.5)
JOB_P09=$(sbatch -J w3P09-fg10g50 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-fg10-grad05" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=1.0 | awk '{print $NF}')
echo "Submitted P09 (v256-fg10-grad05): Job ID ${JOB_P09}"

# 13. w1024-d8-v256-cool-v384 (18 ep: 16 ep @ 256 + 2 ep cooldown @ 384, grad=0.5)
JOB_P10=$(sbatch -J w3P10-cool384 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-cool-v384" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    cooldown_epochs=2 cooldown_num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P10 (v256-cool-v384): Job ID ${JOB_P10}"

# 14. w1024-d8-v256-cool-v512 (18 ep: 16 ep @ 256 + 2 ep cooldown @ 512, grad=0.5)
JOB_P11=$(sbatch -J w3P11-cool512 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-cool-v512" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    cooldown_epochs=2 cooldown_num_visible=512 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P11 (v256-cool-v512): Job ID ${JOB_P11}"

# 15. w1024-d8-v192-fg15-grad05 (21 ep, v192, 91.4% masking, fg=1.5, grad=0.5)
JOB_P12=$(sbatch -J w3P12-v192-fg15 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v192-fg15-grad05" 7338 21 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=192 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=1.5 | awk '{print $NF}')
echo "Submitted P12 (v192-fg15-grad05): Job ID ${JOB_P12}"

# 16. w1024-d8-v320-fg15-grad05 (16 ep, v320, fg=1.5, grad=0.5)
JOB_P13=$(sbatch -J w3P13-v320-fg15 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v320-fg15-grad05" 7338 16 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=320 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=1.5 | awk '{print $NF}')
echo "Submitted P13 (v320-fg15-grad05): Job ID ${JOB_P13}"

# 17. w1152-d7-v256-grad05 (18 ep, w1152, d7, v256, grad=0.5)
JOB_P14=$(sbatch -J w3P14-w1152d7 --time=2:30:00 "${S_TRAIN}" "1.0e/w1152-d7-v256-grad05" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1152 model_kwargs.depth=7 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P14 (w1152-d7-v256-grad05): Job ID ${JOB_P14}"

# 18. w1024-d8-v256-grad075 (18 ep, v256, grad=0.75)
JOB_P15=$(sbatch -J w3P15-grad075 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-grad075" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.75 | awk '{print $NF}')
echo "Submitted P15 (v256-grad075): Job ID ${JOB_P15}"

# 19. w1024-d8-v256-grad025 (18 ep, v256, grad=0.25)
JOB_P16=$(sbatch -J w3P16-grad025 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-grad025" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.25 | awk '{print $NF}')
echo "Submitted P16 (v256-grad025): Job ID ${JOB_P16}"

# 20. w1280-d7-v256-grad05 (16 ep, w1280, d7, v256, grad=0.5)
JOB_P17=$(sbatch -J w3P17-w1280d7 --time=2:30:00 "${S_TRAIN}" "1.0e/w1280-d7-v256-grad05" 7338 16 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=7 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P17 (w1280-d7-v256-grad05): Job ID ${JOB_P17}"

# 21. w1024-d8-v256-dec3-w512 (17 ep, v256, dec_depth=3, dec_dim=512, grad=0.5)
JOB_P18=$(sbatch -J w3P18-dec3w512 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-dec3-w512" 7338 17 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=3 model_kwargs.decoder_embed_dim=512 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P18 (v256-dec3-w512): Job ID ${JOB_P18}"

# 22. w1024-d8-v256-lr4e3 (18 ep, v256, base_lr=0.004, grad=0.5)
JOB_P19=$(sbatch -J w3P19-lr4e3 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-lr4e3" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.004 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P19 (v256-lr4e3): Job ID ${JOB_P19}"

# 23. w1024-d8-v256-warmup3 (18 ep, v256, warmup=3, grad=0.5)
JOB_P20=$(sbatch -J w3P20-warmup3 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256-warmup3" 7338 18 3 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P20 (v256-warmup3): Job ID ${JOB_P20}"

# 24. w960-d8-v256-grad05 (19 ep, w960, d8, v256, grad=0.5)
JOB_P21=$(sbatch -J w3P21-w960d8 --time=2:30:00 "${S_TRAIN}" "1.0e/w960-d8-v256-grad05" 7338 19 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=960 model_kwargs.depth=8 model_kwargs.num_heads=15 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 | awk '{print $NF}')
echo "Submitted P21 (w960-d8-v256-grad05): Job ID ${JOB_P21}"

echo "=========================================================="
echo " All 24 runs successfully submitted to SLURM queue!"
echo "=========================================================="

