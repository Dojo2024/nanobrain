#!/usr/bin/env bash
#SBATCH --job-name=arm-resume
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:30:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# Continue one seed of an arm_1hr.sbatch run from its last checkpoint to a fixed epoch count
# (stop_epoch), then the same evals as arm_1hr.sbatch. Used to compare an arm with its baseline
# at matched epochs when the arm is slower per step.
# usage: sbatch -J <job name> scripts/arm_resume.sbatch <run name> <seed> <source run> <stop epoch> [overrides...]
#   e.g. sbatch -J nb-l1-dhc4-w384-matched-s1 scripts/arm_resume.sbatch l1/dhc4-w384-matched 1 \
#        l1/dhc4-w384 15 base_lr=0.003 ... model_kwargs.hyper_connections=4
#   The overrides must be the source run's own; the model then loads the checkpoint strictly.
#
# The checkpoint and logs are copied into the new run dir, so the source run and its evals stay
# as they are. config.yaml is not copied: train.py refuses to resume under a different config, and
# older run dirs predate config keys added since, so the new run writes its own. The lr schedule
# is keyed to the global step and the optimizer state is restored, so the continuation follows
# the uninterrupted schedule; only the data and mask order after the restart differ.

set -euo pipefail

RUN="$1"
SEED="$2"
SRC="$3"
STOP="$4"
shift 4
cd /admin/home/dojo/nanobrain

set -a
source .env
set +a
export VIRTUAL_ENV=

DATA_ROOT="/data/nanobrain/FOMO300K_images"
OUT_DIR="/data/dojo/nanobrain/output"
NAME="${RUN}/s${SEED}"
RUN_DIR="${OUT_DIR}/${NAME}"
SRC_DIR="${OUT_DIR}/${SRC}/s${SEED}"

if [[ -e "${RUN_DIR}/checkpoint-last.pth" ]]; then
    echo "${RUN_DIR} already has a checkpoint" >&2
    exit 1
fi
mkdir -p "${RUN_DIR}"
cp "${SRC_DIR}/checkpoint-last.pth" "${SRC_DIR}/log.json" "${SRC_DIR}/log.txt" "${RUN_DIR}/"

# train to stop_epoch; the 2 h timeout is only a guard, a run that hits it failed
.venv/bin/python -c "import torch; print('resuming after epoch', torch.load('${RUN_DIR}/checkpoint-last.pth', map_location='cpu', weights_only=False)['epoch'] + 1, 'to ${STOP}')"
timeout --kill-after=2m "2h" \
    .venv/bin/python -m nanobrain.train \
    --overrides \
    name="${NAME}" \
    output_dir="${OUT_DIR}" \
    seed="${SEED}" \
    data_root="${DATA_ROOT}" \
    train_filelist="${DATA_ROOT}/filelist_subsampled.txt" \
    stop_epoch="${STOP}" \
    "$@"

# the default eval on checkpoint-last, then the layers 9+12 readout, as arm_1hr.sbatch
.venv/bin/python -m nanobrain.eval.run \
    "${RUN_DIR}/checkpoint-last.pth" \
    "${RUN_DIR}/eval"

READOUT_DIR="${RUN_DIR}/readout/cat9-12"
.venv/bin/python -m nanobrain.readout \
    "${RUN_DIR}/checkpoint-last.pth" "${READOUT_DIR}/model.pth" --layers 9 12
for task in $(.venv/bin/python -c 'from nanobrain.eval.tasks import DEFAULT_TASKS; print(*DEFAULT_TASKS)'); do
    if ! .venv/bin/python -m nanobrain.eval.run "${READOUT_DIR}/model.pth" "${READOUT_DIR}" --tasks "${task}"; then
        echo "FAILED cat9-12 ${task}" | tee -a "${RUN_DIR}/readout/failed.txt"
    fi
done
rm "${READOUT_DIR}/model.pth"
