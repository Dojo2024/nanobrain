#!/usr/bin/env bash
#
#   Soup / checkpoint eval on the v3 harness: one checkpoint -> all 11 open
#   tasks, in two flavors (final-layer and layered readout) selected by MODE.
#   Usage: sbatch soup_eval.sbatch <RUN_DIR> <MODE>   MODE in: final | layered
#
#SBATCH --job-name=nb-soup-eval
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --output=/admin/home/dojo/nanobrain-v3/slurms/soupeval_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain-v3/slurms/soupeval_%j.err

set -euo pipefail

RUN_DIR="${1:?run dir, e.g. output/soup-champ}"
MODE="${2:-final}"

cd /admin/home/dojo/nanobrain-v3
export VIRTUAL_ENV=
export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export OMP_NUM_THREADS=2

CKPT="${RUN_DIR}/soup-last.pth"
[[ -f "${CKPT}" ]] || { echo "ERROR: ${CKPT} not found" >&2; exit 1; }

RL="null"
NAME="eval"
if [[ "${MODE}" == "layered" ]]; then
    RL="[1,2,4,6,8]"
    NAME="eval-layered"
fi

OUT="${RUN_DIR}/eval-${MODE}"
mkdir -p "${OUT}"

TASKS=$(uv run --no-sync python -c "from brainmarks_smri.tasks import list_tasks; print(*list_tasks())")
for task in ${TASKS}; do
    uv run --no-sync python -m brainmarks_smri vit_mae_3d "${task}" \
        --overrides \
        "imports=[nanobrain.model]" \
        "model_kwargs.checkpoint=${CKPT}" \
        "model_kwargs.readout_layers=${RL}" \
        output_dir="${OUT}" \
        name="${NAME}" || echo "task ${task} failed" >&2
done
echo "=== soup eval done: ${OUT} (${MODE}) ==="
