#!/usr/bin/env bash
#SBATCH --job-name=readout
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# Layer readout sweep on one checkpoint: rewrites checkpoint-last to a
# nanobrain.readout model per readout and runs the unmodified eval on it.
# usage: sbatch -J <job name> scripts/readout_eval.sbatch <run name> <seed> [<readout name> <tasks> <flags...>]
#   e.g. sbatch -J n1n3-readout-s7338 scripts/readout_eval.sbatch n8/lr3e-3-cut40 7338
#   With only <run name> <seed> it runs the sweep at the end of this file. With more arguments it
#   runs that one readout instead: a name, the tasks ("" for the default 8), and nanobrain.readout
#   flags, e.g. ... l1/dhc4-w384-matched 7338 cat6-9-12 "" --layers 6 9 12
#
# Each task runs on its own, so a probe that fails (e.g. the l-bfgs convergence assert on
# wide concatenations) costs that task only. Failures are listed in ${OUT_DIR}/failed.txt.
#
# A requeued job resumes: eval.run skips tasks whose JSON already exists.
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

RUN_DIR="/data/dojo/nanobrain/output/${RUN}/s${SEED}"
OUT_DIR="${RUN_DIR}/readout"
mkdir -p "${OUT_DIR}"

# GPU memory and utilization every 30 s
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 30 \
    > "${OUT_DIR}/gpu_${SLURM_JOB_ID}.csv" &
SMI_PID=$!
trap 'kill ${SMI_PID}' EXIT

DEFAULT_TASKS="$(.venv/bin/python -c 'from nanobrain.eval.tasks import DEFAULT_TASKS; print(*DEFAULT_TASKS)')"

# usage: run_readout <name> <tasks, empty for the default 8> <nanobrain.readout flags...>
run_readout() {
    local name="$1"
    local tasks="${2:-${DEFAULT_TASKS}}"
    shift 2
    local ckpt="${OUT_DIR}/${name}/model.pth"
    echo "[$(date +%T)] readout ${name}: $*"
    .venv/bin/python -m nanobrain.readout "${RUN_DIR}/checkpoint-last.pth" "${ckpt}" "$@"
    for task in ${tasks}; do
        if ! .venv/bin/python -m nanobrain.eval.run "${ckpt}" "${OUT_DIR}/${name}" --tasks "${task}"; then
            echo "[$(date +%T)] FAILED ${name} ${task}"
            echo "${name} ${task}" >> "${OUT_DIR}/failed.txt"
        fi
    done
    rm "${ckpt}"
}

# one readout given on the command line instead of the sweep below
if (( $# )); then
    run_readout "$@"
    exit 0
fi

# 2026-10-03 sweep on baseline/s*: l12 (control), l3, l6, l9, mean3-6-9-12, cat9-12 (layers 9-12),
# cat3-6-9-12, cls-l12. Results in nanobrain-ideas.md section 0.10.

# 2026-10-04 on n8/lr3e-3-cut40/s*: cat6-12, cat9-12 (layers 9 and 12), cls-l12 (N1 and N3).

# 2026-10-05 on a17/batched-nt/s*: cat6-9-12 (layers 6, 9, 12, 1152 wide), the dense half of the
# split readout.

# 2026-10-06 on l1/dhc4-w384-matched/s* and a17/batched-nt/s*: layers 3, 6 and 9 one at a time
# (384 wide), compared to ${RUN_DIR}/eval (layer 12). With hyper-connections a layer is the sum
# of its streams.
run_readout l3 "" --layers 3
run_readout l6 "" --layers 6
run_readout l9 "" --layers 9
echo "[$(date +%T)] done"
