#!/usr/bin/env bash
set -euo pipefail

chmod +x /admin/home/dojo/nanobrain/scripts/eval_seg_boost.sbatch
chmod +x /admin/home/dojo/nanobrain/scripts/train_seg_boost.sbatch

S_EVAL="/admin/home/dojo/nanobrain/scripts/eval_seg_boost.sbatch"
S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_seg_boost.sbatch"

echo "=== Tier 1: Submitting 6 Zero-Pretraining Feature-Extraction Runs ==="
# E1: 1.0e/w1024-d8-v384 + readout_layers=[4,6,8]
sbatch -J segE1-cat468 "${S_EVAL}" "1.0e/w1024-d8-v384" "1.0e/w1024-d8-v384-cat468" 7338 \
    "model_kwargs.readout_layers=[4,6,8]"

# E2: 1.0e/w1024-d8-v384 + readout_layers=[2,4,6,8]
sbatch -J segE2-cat2468 "${S_EVAL}" "1.0e/w1024-d8-v384" "1.0e/w1024-d8-v384-cat2468" 7338 \
    "model_kwargs.readout_layers=[2,4,6,8]"

# E3: 1.0e/w1024-d8-v384 + dense_neighbor_pool=true
sbatch -J segE3-nbr3 "${S_EVAL}" "1.0e/w1024-d8-v384" "1.0e/w1024-d8-v384-nbr3" 7338 \
    "model_kwargs.dense_neighbor_pool=true"

# E4: 1.0e/w1024-d8-v384 + readout_layers=[4,6,8] + dense_neighbor_pool=true
sbatch -J segE4-cat-nbr "${S_EVAL}" "1.0e/w1024-d8-v384" "1.0e/w1024-d8-v384-cat468-nbr3" 7338 \
    "model_kwargs.readout_layers=[4,6,8]" "model_kwargs.dense_neighbor_pool=true"

# E5: 2.0e/w1280-d8 + readout_layers=[4,6,8]
sbatch -J segE5-20e-cat "${S_EVAL}" "2.0e/w1280-d8" "2.0e/w1280-d8-cat468" 7338 \
    "model_kwargs.readout_layers=[4,6,8]"

# E6: 2.0e/w1280-d8 + readout_layers=[4,6,8] + dense_neighbor_pool=true
sbatch -J segE6-20e-catnbr "${S_EVAL}" "2.0e/w1280-d8" "2.0e/w1280-d8-cat468-nbr3" 7338 \
    "model_kwargs.readout_layers=[4,6,8]" "model_kwargs.dense_neighbor_pool=true"


echo "=== Tier 2: Submitting 9 1.0-EFLOP Pretraining + Eval Runs ==="
# Common 1.0e overrides: optimizer=muon muon_impl=fused base_lr=0.003 model_kwargs.norm_target=true

# S1: 1.0e w1024-d8-v384 + shallow decoder (d=2, w=384, 18 epochs)
sbatch -J segS1-10e-dec2 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v384-dec2" 7338 18 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6 \
    model_kwargs.norm_target=true

# S2: 1.0e w1024-d8-v384 + 3D spatial gradient loss (grad_loss_weight=0.5, 15 epochs)
sbatch -J segS2-10e-grad --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v384-grad" 7338 15 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# S3: 1.0e w1024-d8-v384 + dense-context cooldown (13 ep @ v384 + 1 ep @ v1280 = 14 ep)
sbatch -J segS3-10e-cool --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v384-cool" 7338 14 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    cooldown_epochs=1 cooldown_num_visible=1280 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true

# S4: 1.0e w1024-d8-v384-mlp6 (135M, mlp_ratio=6 + v384, 12 epochs)
sbatch -J segS4-10e-mlp6 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v384-mlp6" 7338 12 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.mlp_ratio=6 model_kwargs.norm_target=true

# S5: 1.0e w1280-d8-v384-dec2 (158M, shallow decoder d=2 w=384 -> 13 epochs at 1.0e)
sbatch -J segS5-10e-w1280dec2 --time=2:30:00 "${S_TRAIN}" "1.0e/w1280-d8-v384-dec2" 7338 13 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6 \
    model_kwargs.norm_target=true

# S6: 1.0e w1024-d8-v256 (92% masking, 22 epochs at 1.0e)
sbatch -J segS6-10e-v256 --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-v256" 7338 22 2 100m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=256 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true

# S7: 1.0e Full w1024-d8 Seg Combo (v384 + dec2 + grad + cool 16+1=17 ep + cat[4,6,8]+nbr3 eval)
sbatch -J segS7-10e-combo --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-seg-combo" 7338 17 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    cooldown_epochs=1 cooldown_num_visible=1280 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    "model_kwargs.eval_readout_layers=[4,6,8]" model_kwargs.dense_neighbor_pool=true

# S8: 1.0e Full w1024-d8-mlp6 Seg Combo (13+1=14 ep + cat[4,6,8]+nbr3 eval)
sbatch -J segS8-10e-mlp6combo --time=2:30:00 "${S_TRAIN}" "1.0e/w1024-d8-mlp6-seg-combo" 7338 14 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    cooldown_epochs=1 cooldown_num_visible=1280 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.mlp_ratio=6 \
    model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    "model_kwargs.eval_readout_layers=[4,6,8]" model_kwargs.dense_neighbor_pool=true

# S9: 1.0e Full w1280-d8 Seg Combo (12+1=13 ep + cat[4,6,8]+nbr3 eval)
sbatch -J segS9-10e-w1280combo --time=2:30:00 "${S_TRAIN}" "1.0e/w1280-d8-seg-combo" 7338 13 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    cooldown_epochs=1 cooldown_num_visible=1280 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    "model_kwargs.eval_readout_layers=[4,6,8]" model_kwargs.dense_neighbor_pool=true


echo "=== Tier 3: Submitting 6 2.0-EFLOP Pretraining + Eval Runs ==="
# S10: 2.0e w1280-d8-v384 (20 epochs)
sbatch -J segS10-20e-w1280v384 --time=3:45:00 "${S_TRAIN}" "2.0e/w1280-d8-v384" 7338 20 2 175m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true

# S11: 2.0e w1024-d8-v384 (30 epochs)
sbatch -J segS11-20e-w1024v384 --time=3:45:00 "${S_TRAIN}" "2.0e/w1024-d8-v384" 7338 30 3 175m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true

# S12: 2.0e w1280-d8-v384-dec2 (25 epochs)
sbatch -J segS12-20e-dec2 --time=3:45:00 "${S_TRAIN}" "2.0e/w1280-d8-v384-dec2" 7338 25 2 175m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6 \
    model_kwargs.norm_target=true

# S13: 2.0e w1280-d8-v384-grad (20 epochs)
sbatch -J segS13-20e-grad --time=3:45:00 "${S_TRAIN}" "2.0e/w1280-d8-v384-grad" 7338 20 2 175m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# S14: 2.0e Full w1280-d8 Seg Combo (23+2=25 ep + cat[4,6,8]+nbr3 eval)
sbatch -J segS14-20e-combo --time=4:00:00 "${S_TRAIN}" "2.0e/w1280-d8-seg-combo" 7338 25 2 185m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    cooldown_epochs=2 cooldown_num_visible=1280 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    "model_kwargs.eval_readout_layers=[4,6,8]" model_kwargs.dense_neighbor_pool=true

# S15: 2.0e Full w1280-d8-mlp6 Seg Combo (210M, 18+1=19 ep + cat[4,6,8]+nbr3 eval)
sbatch -J segS15-20e-mlp6combo --time=4:00:00 "${S_TRAIN}" "2.0e/w1280-d8-mlp6-seg-combo" 7338 19 2 185m \
    optimizer=muon muon_impl=fused base_lr=0.003 num_visible=384 \
    cooldown_epochs=1 cooldown_num_visible=1280 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.mlp_ratio=6 \
    model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    "model_kwargs.eval_readout_layers=[4,6,8]" model_kwargs.dense_neighbor_pool=true
