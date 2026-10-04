#!/usr/bin/env bash
#SBATCH --job-name=arm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# One seed of an ablation arm under the 1 h wall time budget, same flow as baseline_1hr.sbatch
# plus extra config overrides, then the default nanobrain eval on checkpoint-last, then the same
# eval on the layers 9+12 readout (readout/cat9-12), then the eval on each EMA checkpoint (eval-ema*).
# usage: sbatch -J <job name> scripts/arm_1hr.sbatch <run name> <seed> [overrides...]
#   e.g. sbatch -J n7 scripts/arm_1hr.sbatch n7/cos14 7338 epochs=14 warmup_epochs=1
#
# --no-requeue: a resumed run would train past the 1 h budget, so a preempted seed stops
# instead, and the guard below refuses to reuse a run dir that already has a checkpoint.
# n-4 excluded on 2026-10-03, ~15 GB host RAM free.

set -euo pipefail

RUN="$1"
SEED="$2"
shift 2
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
    echo "${RUN_DIR} already has a checkpoint; resuming would break the 1 h budget" >&2
    exit 1
fi

# run training with a fixed wall time budget
# timeout exits 124 when it stops training, which is expected
status=0
timeout --kill-after=2m "1h" \
    .venv/bin/python -m nanobrain.train \
    --overrides \
    name="${NAME}" \
    output_dir="${OUT_DIR}" \
    seed="${SEED}" \
    data_root="${DATA_ROOT}" \
    train_filelist="${DATA_ROOT}/filelist_subsampled.txt" \
    "$@" \
    || status=$?
if [[ $status -ne 0 && $status -ne 124 ]]; then
    exit $status
fi

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
