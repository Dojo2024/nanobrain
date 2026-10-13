#!/usr/bin/env bash
#SBATCH --job-name=nb-bmro
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# Whole-scan readouts on brainmarks for one finished run, no training (PROJECT_STATE.md section 8):
# 1. brain_mode=pool on checkpoint-last: the global embedding averages only patches with brain voxels
#    -> <run dir>/brainmarks-brainpool/
# 2. the layers 9+12 readout (nanobrain.readout --layers 9 12): global = mean-pooled layer 9 and
#    layer 12 tokens, concatenated -> <run dir>/brainmarks-cat9-12/
# Both on the 7 whole-scan tasks only; segmentation is unchanged by (1) and not asked for in (2).
# usage: sbatch -J nb-bmro-<arm>-s<seed> experiments/dojo_1h/readout/bm_global_readouts.sbatch <run dir>

set -euo pipefail
RUN_DIR="$1"
cd /admin/home/dojo/nanobrain
export VIRTUAL_ENV=
export TASKS="abide_diagnosis cnp_diagnosis ucsf_idh upenn_survival_1y soop_mrs_poor ixi_age openbhb_age"

bash scripts/brainmarks.sbatch "${RUN_DIR}/checkpoint-last.pth" "${RUN_DIR}" brainmarks-brainpool \
    model_kwargs.brain_mode=pool

# arm_1hr.sbatch deletes the readout model after its FOMO eval, so rebuild it (no new weights)
READOUT="${RUN_DIR}/readout/cat9-12/model.pth"
.venv/bin/python -m nanobrain.readout "${RUN_DIR}/checkpoint-last.pth" "${READOUT}" --layers 9 12
bash scripts/brainmarks.sbatch "${READOUT}" "${RUN_DIR}" brainmarks-cat9-12
rm "${READOUT}"
