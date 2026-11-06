#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_TRAIN="/admin/home/dojo/nanobrain/scripts/train_scale_4_8h.sbatch"
chmod +x "${S_TRAIN}"

echo "=========================================================="
echo " Submitting Part 2 (Jobs 5-8): 2.0 EFLOP Nanobrain Scaling"
echo " Strict Last-Layer Evaluation Protocol (No Intermediate Layers)"
echo " Cluster Guardrails: cpus-per-task=8, mem=48G, requeue enabled"
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
# 5. w2e-05-w1024d12: Deeper Hierarchical Capacity Frontier (22 epochs)
# w1024-d12 (12 layers) allows deeper hierarchical representations at 2.0 EFLOPs
# ------------------------------------------------------------
submit_2e "w2e-05-w1024d12" "2.0e/w1024-d12-v384-grad05-nbr" 7338 22 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=12 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 6. w2e-06-v512-dense: Maximum Resolution Density (84% Masking) (20 epochs)
# 512 visible tokens captures micro-structures (cranial nerves, vessels, subtle lesions)
# ------------------------------------------------------------
submit_2e "w2e-06-v512" "2.0e/w1024-d8-v512-grad05-nbr" 7338 20 2 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0032 num_visible=512 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 7. w2e-07-curr-res: Dynamic Token Resolution Curriculum (V=256 -> V=448) (26 epochs)
# Learns broad global anatomy early, sharpening to fine boundaries in late epochs
# ------------------------------------------------------------
submit_2e "w2e-07-curr" "2.0e/curr-v256-to-v448-lr35-g50-nbr" 7338 26 3 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0035 num_visible=448 \
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

# ------------------------------------------------------------
# 8. w2e-08-w1280d8: Ultra-Wide Capacity Frontier (18 epochs)
# Width=1280 massive channel capacity with depth=8 at 2.0 EFLOPs
# ------------------------------------------------------------
submit_2e "w2e-08-w1280d8" "2.0e/w1280-d8-v384-grad05-nbr" 7338 18 2 "2.5h" "04:00:00" \
    optimizer=muon muon_impl=fused base_lr=0.0030 num_visible=384 \
    model_kwargs.embed_dim=1280 model_kwargs.depth=8 model_kwargs.num_heads=16 \
    model_kwargs.norm_target=true model_kwargs.grad_loss_weight=0.5 \
    model_kwargs.dense_neighbor_pool=true

echo "=========================================================="
echo " Part 2 (Jobs 5-8) successfully queued into SLURM!"
echo "=========================================================="
