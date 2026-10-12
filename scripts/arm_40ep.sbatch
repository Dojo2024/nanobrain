#!/usr/bin/env bash
#SBATCH --job-name=arm-40ep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# One seed of an arm trained from scratch for the full 40-epoch schedule (no 1 h cut, so the lr
# decays to the end), then the arm evals on the final checkpoint and last-layer evals on the
# periodic checkpoints at epochs 10, 20 and 30 (checkpoint_period 10). The 1 h runs follow the
# first ~13 epochs of the same schedule, so the periodic evals show whether 1 h rankings hold.
# No resume: a preempted run stops and is rerun from scratch.
# usage: sbatch -J <job name> scripts/arm_40ep.sbatch <run name> <seed> [overrides...]
#   e.g. sbatch -J nb-e40-adopted-w512-s1 scripts/arm_40ep.sbatch e40/adopted-w512 1 \
#        base_lr=0.003 optimizer=muon muon_impl=batched norm_target=true \
#        model_kwargs.embed_dim=512 model_kwargs.num_heads=8
# The 7 h timeout is only a guard: a run that hits it failed.

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
    echo "${RUN_DIR} already has a checkpoint" >&2
    exit 1
fi

# train all 40 epochs
timeout --kill-after=2m "7h" \
    .venv/bin/python -m nanobrain.train \
    --overrides \
    name="${NAME}" \
    output_dir="${OUT_DIR}" \
    seed="${SEED}" \
    data_root="${DATA_ROOT}" \
    train_filelist="${DATA_ROOT}/filelist_subsampled.txt" \
    "$@"

# default eval on the final checkpoint, ~6 min
.venv/bin/python -m nanobrain.eval.run \
    "${RUN_DIR}/checkpoint-last.pth" \
    "${RUN_DIR}/eval"

# layers 9 and 12 concatenated, as arm_1hr.sbatch, ~12 min
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

# last layer at epochs 10, 20, 30 (checkpoint-000NN.pth is saved after 0-indexed epoch NN), ~6 min each
for epoch in 10 20 30; do
    if ! .venv/bin/python -m nanobrain.eval.run \
        "${RUN_DIR}/checkpoint-$(printf '%05d' $((epoch - 1))).pth" "${RUN_DIR}/eval-ep${epoch}"; then
        echo "FAILED eval ep${epoch}" | tee -a "${RUN_DIR}/readout/failed.txt"
    fi
done
