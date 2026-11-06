#!/usr/bin/env bash
#SBATCH --job-name=nb-seg-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

set -euo pipefail

SRC_RUN="$1"
DEST_RUN="$2"
SEED="$3"
shift 3

cd /admin/home/dojo/nanobrain
export VIRTUAL_ENV=
export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export PYTHONPATH=/admin/home/dojo/brainmarks-smri/src:/admin/home/dojo/nanobrain/src
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="/data/dojo/nanobrain/output/isoflop"
SRC_DIR="${OUT_DIR}/${SRC_RUN}/s${SEED}"
DEST_DIR="${OUT_DIR}/${DEST_RUN}/s${SEED}"
CKPT="${SRC_DIR}/checkpoint-last.pth"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: ${CKPT} not found" >&2
    exit 1
fi

mkdir -p "${DEST_DIR}"
if [[ -f "${SRC_DIR}/log.json" && ! -f "${DEST_DIR}/log.json" ]]; then
    cp "${SRC_DIR}/log.json" "${DEST_DIR}/log.json"
fi
if [[ -f "${SRC_DIR}/config.yaml" && ! -f "${DEST_DIR}/config.yaml" ]]; then
    cp "${SRC_DIR}/config.yaml" "${DEST_DIR}/config.yaml"
fi

if [[ -d "${DEST_DIR}/brainmarks" && $(ls -1 "${DEST_DIR}/brainmarks"/*.json 2>/dev/null | wc -l) -ge 19 ]]; then
    echo "${DEST_DIR} already fully evaluated across all 19 tasks."
    exit 0
fi

TASKS=$(.venv/bin/python -c "from brainmarks_smri.tasks import list_tasks; print(*list_tasks())")
for task in ${TASKS}; do
    if [[ -f "${DEST_DIR}/brainmarks/${task}.json" ]]; then
        echo "Task ${task} already evaluated, skipping..."
        continue
    fi
    if ! .venv/bin/python -m brainmarks_smri vit_mae_3d "${task}" \
        --overrides \
        "imports=[nanobrain.model]" \
        model_kwargs.checkpoint="${CKPT}" \
        "$@" \
        output_dir="${DEST_DIR}" \
        name=brainmarks; then
        echo "FAILED ${task}" | tee -a "${DEST_DIR}/brainmarks-failed.txt"
    fi
done

