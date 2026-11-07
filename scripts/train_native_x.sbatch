#!/usr/bin/env bash
#
#   Native-resolution screening arms, all on Mihir's published r12 source
#   (the level-set head only exists in run_sub_8eb1eb614b's snapshot):
#     r12     - exact leader replication, anchor/baseline (first run 577945 is this)
#     nx      - r12 + our v3 loss stack (norm_target, grad loss, fg loss)
#     nxnl    - nx with the level head OFF (ablation: attributes the gain)
#     r12long - r12 at 24500 steps: fills the ~5 idle minutes of the 55 min budget
#
#   Usage: sbatch train_native_x.sbatch <ARM> [SEED]
#
#SBATCH --job-name=nb-native
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=3:00:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --output=slurms/native_%j.out
#SBATCH --error=slurms/native_%j.err

set -euo pipefail

cd /admin/home/dojo/nanobrain-r12

export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export OMP_NUM_THREADS=2

ARM="${1:?arm: r12 | nx | nxnl | r12long}"
SEED="${2:-7338}"

case "${ARM}" in
r12)     CONFIG=src/nanobrain/config/r12_nat_lvl_mf.yaml;   NAME="native-r12-s${SEED}" ;;
nx)      CONFIG=src/nanobrain/config/native_x.yaml;         NAME="native-nx-s${SEED}" ;;
nxnl)    CONFIG=src/nanobrain/config/native_x_nolevel.yaml; NAME="native-nxnl-s${SEED}" ;;
r12long) CONFIG=src/nanobrain/config/r12_long.yaml;         NAME="native-r12long-s${SEED}" ;;
*)       echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac
RUN_DIR="output/${NAME}"
mkdir -p "${RUN_DIR}"

uv run --no-sync python -m nanobrain.train \
    --config "${CONFIG}" \
    --overrides name="${NAME}" seed="${SEED}" output_dir=output

uv run --no-sync python -m brainmarks_smri vit_mae_3d open \
    --overrides \
    "imports=[nanobrain.model]" \
    model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
    output_dir="${RUN_DIR}" \
    name=eval

echo "=== native ${ARM} s${SEED} done (submit manually: uv run python scripts/submit.py ${RUN_DIR}) ==="
