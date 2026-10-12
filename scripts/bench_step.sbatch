#!/usr/bin/env bash
#SBATCH --job-name=nb-bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# Step time and peak memory of plain vs hyper-connection variants (scripts/bench_step.py), each
# in its own process so compile caches and memory start clean. Runs from this worktree's src.
# usage: sbatch scripts/bench_step.sbatch

set -uo pipefail
cd /admin/home/dojo/nanobrain-hc
export PYTHONPATH="$PWD/src"
PY=/admin/home/dojo/nanobrain/.venv/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY scripts/bench_step.py plain
$PY scripts/bench_step.py naive-dhc4 hyper_connections=4 hc_dynamic=true
$PY scripts/bench_step.py dhc4 hyper_connections=4 hc_dynamic=true
$PY scripts/bench_step.py shc4 hyper_connections=4 hc_dynamic=false
grep -h RESULT "/admin/home/dojo/nanobrain/slurms/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" || true
