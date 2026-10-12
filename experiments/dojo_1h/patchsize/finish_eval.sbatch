#!/usr/bin/env bash
#SBATCH --job-name=nb-ps-finish
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-1,n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# Finish the evals of a 1 h run whose arm_1hr.sbatch eval stopped at a failed probe fit (ps/p16-w768:
# l-bfgs did not converge on hippocampus, so set -e skipped everything after it):
# 1. the FOMO tasks without a result in <run dir>/eval, one task per call so a failed fit costs only it
# 2. the layers 9+12 readout, as in arm_1hr.sbatch
# 3. brainmarks on checkpoint-last (scripts/brainmarks.sbatch, run in this job)
# usage: sbatch -J <job name> experiments/dojo_1h/patchsize/finish_eval.sbatch <run dir>

set -euo pipefail
RUN_DIR="$1"
cd /admin/home/dojo/nanobrain
set -a
source .env
set +a
export VIRTUAL_ENV=

TASKS="$(.venv/bin/python -c 'from nanobrain.eval.tasks import DEFAULT_TASKS; print(*DEFAULT_TASKS)')"
for task in ${TASKS}; do
    [[ -e "${RUN_DIR}/eval/${task}.json" ]] && continue
    if ! .venv/bin/python -m nanobrain.eval.run "${RUN_DIR}/checkpoint-last.pth" "${RUN_DIR}/eval" --tasks "${task}"; then
        echo "FAILED eval ${task}" | tee -a "${RUN_DIR}/eval-failed.txt"
    fi
done

READOUT_DIR="${RUN_DIR}/readout/cat9-12"
.venv/bin/python -m nanobrain.readout "${RUN_DIR}/checkpoint-last.pth" "${READOUT_DIR}/model.pth" --layers 9 12
for task in ${TASKS}; do
    if ! .venv/bin/python -m nanobrain.eval.run "${READOUT_DIR}/model.pth" "${READOUT_DIR}" --tasks "${task}"; then
        echo "FAILED cat9-12 ${task}" | tee -a "${RUN_DIR}/readout/failed.txt"
    fi
done
rm "${READOUT_DIR}/model.pth"

bash scripts/brainmarks.sbatch "${RUN_DIR}/checkpoint-last.pth" "${RUN_DIR}" brainmarks
