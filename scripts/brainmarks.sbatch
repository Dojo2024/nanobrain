#!/usr/bin/env bash
#SBATCH --job-name=brainmarks
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# brainmarks-smri on one nanobrain checkpoint: the label/grid alignment check, then every task with
# experiments/brainmarks (adapter + launcher), nanobrain's venv and the unmodified brainmarks clone.
# Results: <out dir>/<name>/<task>.json. Extra args are brainmarks config overrides for every task.
# TASKS="a b c" picks a subset of tasks.
# usage: sbatch -J <job name> scripts/brainmarks.sbatch <checkpoint> <out dir> <name> [overrides...]
#   e.g. smoke test: ... <run dir>/checkpoint-last.pth <run dir> brainmarks-smoke max_per_split=12

set -euo pipefail

CKPT="$1"
OUT="$2"
NAME="$3"
shift 3

cd /admin/home/dojo/nanobrain
export VIRTUAL_ENV=
export DATA_ROOT=/data/brainmarks-smri
export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export PYTHONPATH=/admin/home/dojo/brainmarks-smri/src:/admin/home/dojo/nanobrain/experiments/brainmarks
# 8 loader workers x 2 threads = the 16 cpus
export OMP_NUM_THREADS=2

.venv/bin/python experiments/brainmarks/check_alignment.py "${CKPT}"

# TASKS (space separated) runs a subset, so one checkpoint can be split across jobs
TASKS="${TASKS:-$(.venv/bin/python -c 'from brainmarks_smri.tasks import list_tasks; print(*list_tasks())')}"
for task in ${TASKS}; do
    if ! .venv/bin/python experiments/brainmarks/run.py nanobrain "${task}" --overrides \
        "model_kwargs.checkpoint=${CKPT}" "output_dir=${OUT}" "name=${NAME}" "$@"; then
        echo "FAILED ${task}" | tee -a "${OUT}/${NAME}-failed.txt"
    fi
done
