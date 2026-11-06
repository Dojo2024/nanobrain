#!/usr/bin/env bash
set -euo pipefail

cd /admin/home/dojo/nanobrain
chmod +x scripts/reeval_dev.sbatch

# Top 16 checkpoints across our 2.0e, 4.0e, and completed 8.0e runs:
# 10 best 2.0e checkpoints + 4 best 4.0e checkpoints + 2 completed 8.0e checkpoints
RUNS=(
    # --- Top 10 2.0 EFLOP Checkpoints ---
    "2.0e/w1024-d8-v384-dp05-grad05-fg025-nbr/s7338:2e-w1024-d8-v384-dp05-g50-fg025-nbr"
    "2.0e/w1024-d12-v384-grad05-nbr/s7338:2e-w1024-d12-v384-g50-nbr"
    "2.0e/w1024-d8-v384-dp05-g50-fg025-pyr-bmask/s7338:2e-w1024-d8-v384-dp05-g50-fg025-pyr-bmask"
    "2.0e/w1024-d8-v320-dp05-grad05-fg025-nbr/s7338:2e-w1024-d8-v320-dp05-g50-fg025-nbr"
    "2.0e/w1024-d8-v384-dp05-grad060-fg030-nbr/s7338:2e-w1024-d8-v384-dp05-g60-fg030-nbr"
    "2.0e/w1024-d8-v256-dp05-grad05-fg025-nbr/s7338:2e-w1024-d8-v256-dp05-g50-fg025-nbr"
    "2.0e/curr-d14-v256-to-v384-dp04-g50-fg025-nbr/s7338:2e-curr-d14-v256-to-v384-dp04-g50-fg025-nbr"
    "2.0e/w1024-d8-v384-grad05-fg025-nbr/s7338:2e-w1024-d8-v384-g50-fg025-nbr"
    "2.0e/curr-v224-to-v448-lr35-g50-fg025-nbr/s7338:2e-curr-v224-to-v448-lr35-g50-fg025-nbr"
    "2.0e/w1024-d10-v384-dp05-grad05-fg025-nbr/s7338:2e-w1024-d10-v384-dp05-g50-fg025-nbr"
    # --- Top 4 4.0 EFLOP Checkpoints ---
    "4.0e/curr-d10-v256-to-v384-dp05-g50-fg025-nbr/s7338:4e-curr-d10-v256-to-v384-dp05-g50-fg025-nbr"
    "4.0e/w1024-d14-v384-dp04-grad05-fg025-nbr/s7338:4e-w1024-d14-v384-dp04-g50-fg025-nbr"
    "4.0e/w1024-d16-v384-dp04-grad05-fg025-nbr/s7338:4e-w1024-d16-v384-dp04-g50-fg025-nbr"
    "4.0e/w1280-d10-v384-dp06-grad05-fg025-nbr/s7338:4e-w1280-d10-v384-dp06-g50-fg025-nbr"
    # --- Top 2 Completed 8.0 EFLOP Checkpoints ---
    "8.0e/w1024-d14-v384-dp04-grad05-fg025-nbr/s7338:8e-w1024-d14-v384-dp04-g50-fg025-nbr"
    "8.0e/w1280-d10-v384-dp06-grad05-fg025-nbr/s7338:8e-w1280-d10-v384-dp06-g50-fg025-nbr"
)

for entry in "${RUNS[@]}"; do
    run_rel="${entry%%:*}"
    job_name="${entry##*:}"
    sbatch -J "${job_name}" scripts/reeval_dev.sbatch "${run_rel}"
done
