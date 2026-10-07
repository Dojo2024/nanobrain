#!/usr/bin/env bash
#SBATCH --job-name=arm-epochs
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:15:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# One seed of an ablation arm trained from scratch to a fixed epoch count (stop_epoch) instead of
# the 1 h wall time, then the same evals as arm_1hr.sbatch. For arms slower per step than their
# baseline, compared at matched epochs; the lr schedule still spans `epochs`, so it follows the
# baseline's own (cut) schedule. Replaces running arm_1hr.sbatch and then arm_resume.sbatch.
# usage: sbatch -J <job name> scripts/arm_epochs.sbatch <run name> <seed> <stop epoch> [overrides...]
#   e.g. sbatch -J nb-l1-mhc4-adamw-w384-s1 scripts/arm_epochs.sbatch l1/mhc4-adamw-w384 1 14 \
#        base_lr=0.003 norm_target=true model_kwargs.mhc=4
# The 2.5 h timeout is only a guard: a run that hits it failed.

set -euo pipefail

RUN="$1"
SEED="$2"
STOP="$3"
shift 3
cd /admin/home/dojo/nanobrain

set -a
source .env
set +a
export VIRTUAL_ENV=

DATA_ROOT="/data/nanobrain/FOMO300K_images"
OUT_DIR="/data/dojo/nanobrain/output"
NAME="${RUN}/s${SEED}"
RUN_DIR="${OUT_DIR}/${NAME}"

if [[ -e "${RUN_DIR}/checkpoint-last.pth" ]]; then
    echo "${RUN_DIR} already has a checkpoint" >&2
    exit 1
fi

# train to stop_epoch
timeout --kill-after=2m "2.5h" \
    .venv/bin/python -m nanobrain.train \
    --overrides \
    name="${NAME}" \
    output_dir="${OUT_DIR}" \
    seed="${SEED}" \
    data_root="${DATA_ROOT}" \
    train_filelist="${DATA_ROOT}/filelist_subsampled.txt" \
    stop_epoch="${STOP}" \
    "$@"

# run evals if successful, ~6 min
.venv/bin/python -m nanobrain.eval.run \
    "${RUN_DIR}/checkpoint-last.pth" \
    "${RUN_DIR}/eval"

# second score: layers 9 and 12 concatenated (N1, nanobrain-ideas.md section 0.10), ~12 min.
# Each task runs on its own so a failed probe fit costs that task only.
READOUT_DIR="${RUN_DIR}/readout/cat9-12"
.venv/bin/python -m nanobrain.readout \
    "${RUN_DIR}/checkpoint-last.pth" "${READOUT_DIR}/model.pth" --layers 9 12
for task in $(.venv/bin/python -c 'from nanobrain.eval.tasks import DEFAULT_TASKS; print(*DEFAULT_TASKS)'); do
    if ! .venv/bin/python -m nanobrain.eval.run "${READOUT_DIR}/model.pth" "${READOUT_DIR}" --tasks "${task}"; then
        echo "FAILED cat9-12 ${task}" | tee -a "${RUN_DIR}/readout/failed.txt"
    fi
done
rm "${READOUT_DIR}/model.pth"

# EMA checkpoints (ema_decays), last layer only, ~6 min each, into eval-ema<decay>
for ckpt in "${RUN_DIR}"/checkpoint-ema*.pth; do
    [[ -e "${ckpt}" ]] || continue
    name="$(basename "${ckpt}" .pth)"
    name="${name#checkpoint-}"
    if ! .venv/bin/python -m nanobrain.eval.run "${ckpt}" "${RUN_DIR}/eval-${name}"; then
        echo "FAILED eval ${name}" | tee -a "${RUN_DIR}/readout/failed.txt"
    fi
done
