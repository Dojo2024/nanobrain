#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_seg_boost.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting Wave 6: Next 3 1.0 EFLOP Nanobrain Experiments"
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
# 1. w6-01-v448dec3: Native V=448 + Shallow Decoder (Dec=3, Dim=384) + Gradient Loss
# Forces 8th layer to natively represent high-frequency boundaries
# Budget: 12 epochs @ 448 = 1.0 EFLOP
# ------------------------------------------------------------
submit_train "w6-01-v448dec3" "1.0e/w1024-d8-v448-dec3-g50" 7338 12 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=3 model_kwargs.decoder_embed_dim=384 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# 2. w6-02-c256to448: Resolution Curriculum (13 ep @ 256 + 3 ep @ 448 cooldown)
# Merges deep global semantic abstractions with high-resolution boundary tokens
# Budget: 16 total epochs = 1.0 EFLOP
# ------------------------------------------------------------
submit_train "w6-02-c256to448" "1.0e/curr-v256-to-v448-lr35-g50" 7338 16 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=256 \
    cooldown_epochs=3 cooldown_num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5

# ------------------------------------------------------------
# 3. w6-03-v384fg05dec3: V=384 + Foreground Weighting + Shallow Decoder + Gradient Loss
# Targets brain lesion boundaries and eliminates background voxel over-fitting
# Budget: 15 epochs @ 384 = 1.0 EFLOP
# ------------------------------------------------------------
submit_train "w6-03-v384fg05dec3" "1.0e/w1024-d8-v384-fg05-dec3-g50" 7338 15 2 105m \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.decoder_depth=3 model_kwargs.decoder_embed_dim=384 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 model_kwargs.fg_loss_weight=0.5

echo "=========================================================="
echo " All 3 Nanobrain 1.0 EFLOP experiments successfully queued into SLURM!"
echo "=========================================================="

