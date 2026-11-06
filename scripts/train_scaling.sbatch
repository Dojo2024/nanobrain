#!/usr/bin/env bash
#SBATCH --job-name=nb-scale
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-3,n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

set -euo pipefail

RUN="$1"
SEED="$2"
shift 2

cd /admin/home/dojo/nanobrain
export VIRTUAL_ENV=
export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export PYTHONPATH=/admin/home/dojo/brainmarks-smri/src:/admin/home/dojo/nanobrain/src
export OMP_NUM_THREADS=2

OUT_DIR="/data/dojo/nanobrain/output/scaling_4.4eflop"
NAME="${RUN}/s${SEED}"
RUN_DIR="${OUT_DIR}/${NAME}"

if [[ -e "${RUN_DIR}/checkpoint-last.pth" ]]; then
    echo "${RUN_DIR} already has a checkpoint; refusing to overwrite" >&2
    exit 1
fi

# 4-hour pretraining (4.4 EFLOP regime, 4x of the 1h burst)
status=0
timeout --kill-after=2m "4h" \
    .venv/bin/python -m nanobrain.train \
    --overrides \
    name="${NAME}" \
    output_dir="${OUT_DIR}" \
    seed="${SEED}" \
    epochs=36 \
    warmup_epochs=4 \
    data_root="/data/brainmarks-smri/fomo300k" \
    train_filelist="src/nanobrain/config/fomo300k_images.txt" \
    "$@" \
    || status=$?
if [[ $status -ne 0 && $status -ne 124 ]]; then
    exit $status
fi

if [[ ! -f "${RUN_DIR}/checkpoint-last.pth" ]]; then
    echo "ERROR: ${RUN_DIR}/checkpoint-last.pth not found after pretraining" >&2
    exit 1
fi

# All 19 brainmarks-smri evals directly using the model contract
TASKS=$(.venv/bin/python -c "from brainmarks_smri.tasks import list_tasks; print(*list_tasks())")
for task in ${TASKS}; do
    if ! .venv/bin/python -m brainmarks_smri vit_mae_3d "${task}" \
        --overrides \
        "imports=[nanobrain.model]" \
        model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
        output_dir="${RUN_DIR}" \
        name=brainmarks; then
        echo "FAILED ${task}" | tee -a "${RUN_DIR}/brainmarks-failed.txt"
    fi
done
