#!/usr/bin/env bash
#
#   Sweep launcher. NOT RUN YET - prints the sbatch plan; the sbatch calls at
#   the bottom stay commented. To launch: uncomment them and run this file.
#
set -euo pipefail
cd /admin/home/dojo/nanobrain-sweep

echo "iso-flop sweep: $(ls configs/iso/*.yaml | wc -l) configs x 2 seeds = $(( $(ls configs/iso/*.yaml | wc -l) * 2 )) runs"
for cfg in configs/iso/*.yaml; do
    rel="iso/$(basename "$cfg" .yaml)"
    tl=$(python3 -c "import yaml; print(yaml.safe_load(open('$cfg'))['time_limit'])")
    wall=$(printf '%02d:%02d:00' $(( (tl + 45) / 60 )) $(( (tl + 45) % 60 )))
    for seed in 7338 1; do
        echo "sbatch --time=${wall} -J iso scripts/sweep_run.sbatch ${rel} ${seed}"
        # uncomment to submit:
        # sbatch --time=${wall} -J iso scripts/sweep_run.sbatch "${rel}" "${seed}"
    done
done
