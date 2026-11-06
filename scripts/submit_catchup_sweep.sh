#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain

S_EVAL="/admin/home/dojo/nanobrain/scripts/eval_seg_boost.sbatch"
chmod +x "${S_EVAL}"

echo "=========================================================="
echo " Submitting Catch-Up Evaluation Sweep (Finishing Missing Tasks)"
echo " Guardrails: cpus-per-task=8, mem=48G"
echo "=========================================================="

submit_eval() {
    local job_name="$1"
    local src_run="$2"
    local dest_run="$3"
    shift 3
    local job_id
    job_id=$(sbatch -J "${job_name}" --time=00:35:00 "${S_EVAL}" "${src_run}" "${dest_run}" 7338 "$@" | awk '{print $NF}')
    echo "Submitted ${job_name}: Job ID ${job_id}"
}

# 1. w1024-d8-v384-grad-cat12468-fgpool (18/19 done)
submit_eval "fix-cat12468fg" "1.0e/w1024-d8-v384-grad" "1.0e/w1024-d8-v384-grad-cat12468-fgpool" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 2. w1024-d8-v384-fg-grad-cat12468-spp (18/19 done)
submit_eval "fix-v384fgspp" "1.0e/w1024-d8-v384-fg-grad" "1.0e/w1024-d8-v384-fg-grad-cat12468-spp" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true"

# 3. w1024-d8-v256-fg15-grad05 (18/19 done)
submit_eval "fix-v256fg15" "1.0e/w1024-d8-v256-fg15-grad05" "1.0e/w1024-d8-v256-fg15-grad05" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 4. w1024-d8-v320-fg15-grad05 (18/19 done)
submit_eval "fix-v320fg15" "1.0e/w1024-d8-v320-fg15-grad05" "1.0e/w1024-d8-v320-fg15-grad05" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 5. w1280-d7-v256-grad05 (18/19 done)
submit_eval "fix-w1280d7" "1.0e/w1280-d7-v256-grad05" "1.0e/w1280-d7-v256-grad05" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 6. w1024-d8-v256-fg10-grad05 (17/19 done, record nerve dice 0.0684)
submit_eval "fix-v256fg10" "1.0e/w1024-d8-v256-fg10-grad05" "1.0e/w1024-d8-v256-fg10-grad05" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 7. w1024-d8-v256-dec3-w512 (17/19 done)
submit_eval "fix-v256dec3" "1.0e/w1024-d8-v256-dec3-w512" "1.0e/w1024-d8-v256-dec3-w512" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 8. w1024-d8-v256-warmup3 (17/19 done, record age 0.8939)
submit_eval "fix-v256warm3" "1.0e/w1024-d8-v256-warmup3" "1.0e/w1024-d8-v256-warmup3" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 9. w1024-d8-v320-grad-cat12468-spp (17/19 done)
submit_eval "fix-v320cat12" "1.0e/w1024-d8-v320-grad" "1.0e/w1024-d8-v320-grad-cat12468-spp" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true"

# 10. w1024-d8-v320-grad-slam (17/19 done)
submit_eval "fix-v320slam" "1.0e/w1024-d8-v320-grad" "1.0e/w1024-d8-v320-grad-slam" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 11. w1024-d8-v256-grad-cat12468-spp (17/19 done)
submit_eval "fix-v256cat12" "1.0e/w1024-d8-v256-grad" "1.0e/w1024-d8-v256-grad-cat12468-spp" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true"

# 12. w1024-d8-v256-grad-slam (17/19 done)
submit_eval "fix-v256slam" "1.0e/w1024-d8-v256-grad" "1.0e/w1024-d8-v256-grad-slam" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 13. w1024-d8-v256-fg20 (16/19 done)
submit_eval "fix-v256fg20" "1.0e/w1024-d8-v256-fg20" "1.0e/w1024-d8-v256-fg20" \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true"

# 14. w1024-d8-v320-grad-fgpool (16/19 done)
submit_eval "fix-v320fgp" "1.0e/w1024-d8-v320-grad" "1.0e/w1024-d8-v320-grad-fgpool" \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 15. w1024-d8-v256-grad075 (15/19 done)
submit_eval "fix-v256g75" "1.0e/w1024-d8-v256-grad075" "1.0e/w1024-d8-v256-grad075" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 16. w1024-d8-v256-grad025 (13/19 done, record age 0.8993)
submit_eval "fix-v256g25" "1.0e/w1024-d8-v256-grad025" "1.0e/w1024-d8-v256-grad025" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 17. w1024-d8-v224-grad (15/19 done)
submit_eval "fix-v224g" "1.0e/w1024-d8-v224-grad" "1.0e/w1024-d8-v224-grad" \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true"

# 18. w1024-d8-v256-fg15-grad035 (15/19 done)
submit_eval "fix-v256fg15g35" "1.0e/w1024-d8-v256-fg15-grad035" "1.0e/w1024-d8-v256-fg15-grad035" \
    "model_kwargs.readout_layers=[2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=true" \
    "model_kwargs.decouple_readout=true"

# 19. w1152-d7-v256-grad05 (15/19 done)
submit_eval "fix-w1152d7" "1.0e/w1152-d7-v256-grad05" "1.0e/w1152-d7-v256-grad05" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

# 20. w1024-d8-v256-lr4e3 (14/19 done, record age 0.8953)
submit_eval "fix-v256lr4e3" "1.0e/w1024-d8-v256-lr4e3" "1.0e/w1024-d8-v256-lr4e3" \
    "model_kwargs.readout_layers=[1,2,4,6,8]" \
    "model_kwargs.dense_neighbor_pool=pyramid" \
    "model_kwargs.decouple_readout=true" \
    "model_kwargs.brain_mask_pool=true"

echo "=========================================================="
echo " All catch-up evaluations successfully submitted!"
echo "=========================================================="

