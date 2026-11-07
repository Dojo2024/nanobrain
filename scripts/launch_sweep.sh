#!/usr/bin/env bash
#
#   Sweep launcher. Submits one job per (config, seed) that isn't already in
#   SKIP (still queued or finished from a previous partial launch). Each job
#   is named nb-<config>-s<seed>, so squeue is self-describing.
#
set -euo pipefail
cd /admin/home/dojo/nanobrain-sweep

# jobs already alive from the pre-rename launch: base-e768-d12 s7338/s1, heads16-e768-d12 s7338
SKIP="
heads16-e768-d12|1
heads8-e768-d12|7338
heads8-e768-d12|1
mask-nv1152|7338
mask-nv1152|1

n_cfg=$(ls configs/iso/*.yaml | wc -l)
echo "iso-flop sweep: $n_cfg configs x 2 seeds = $(( n_cfg * 2 )) runs (skipping live ones)"

for cfg in configs/iso/*.yaml; do
    rel="iso/$(basename "$cfg" .yaml)"
    tl=$(python3 -c "import yaml; print(yaml.safe_load(open('$cfg'))['time_limit'])")
    wall=$(printf '%02d:%02d:00' $(( (tl + 45) / 60 )) $(( (tl + 45) % 60 )))
    for seed in 7338 1; do
        if grep -qx "${rel#iso/}|${seed}" <<<"$SKIP"; then
            echo "skip (already live): ${rel#iso/} s${seed}"
            continue
        fi
        sbatch --time=${wall} -J "nb-${rel#iso/}-s${seed}" scripts/sweep_run.sbatch "${rel}" "${seed}"
    done
done
