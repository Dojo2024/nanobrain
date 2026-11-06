#!/usr/bin/env bash
#SBATCH --job-name=nb-layer
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=main
#SBATCH --account=idle
#SBATCH --qos=low
#SBATCH --no-requeue
#SBATCH --exclude=n-3,n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

set -euo pipefail

MODEL_REL_PATH="$1"  # e.g. "ps8/w384-d20"
SEED="${2:-7338}"

cd /admin/home/dojo/nanobrain
export VIRTUAL_ENV=
export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export PYTHONPATH=/admin/home/dojo/brainmarks-smri/src:/admin/home/dojo/nanobrain/src
export OMP_NUM_THREADS=2

OUT_DIR="/data/dojo/nanobrain/output/new_regime"
RUN_DIR="${OUT_DIR}/${MODEL_REL_PATH}/s${SEED}"
CKPT="${RUN_DIR}/checkpoint-last.pth"

if [[ ! -f "${CKPT}" ]]; then
    echo "ERROR: Checkpoint ${CKPT} does not exist" >&2
    exit 1
fi

# Detect model depth
DEPTH=$(.venv/bin/python -c "import torch; ckpt = torch.load('${CKPT}', map_location='cpu', weights_only=True); print(ckpt['args'].get('model_kwargs', {}).get('depth', 12))")
echo "Starting layer sweep for ${MODEL_REL_PATH} s${SEED} (Depth: ${DEPTH})"

# Copy/link final layer results if they already exist in brainmarks/
FINAL_LAYER_DIR="${RUN_DIR}/readout/l${DEPTH}"
mkdir -p "${FINAL_LAYER_DIR}"
if [[ -d "${RUN_DIR}/brainmarks" ]]; then
    echo "Linking existing final layer brainmarks results to readout/l${DEPTH}..."
    for f in "${RUN_DIR}"/brainmarks/*.json; do
        if [[ -f "$f" ]]; then
            cp -n "$f" "${FINAL_LAYER_DIR}/" || true
        fi
    done
fi

TASKS=$(.venv/bin/python -c "from brainmarks_smri.tasks import list_tasks; print(*list_tasks())")

# Sweep all individual layers 1 to DEPTH-1 (DEPTH already handled)
for layer in $(seq 1 $((DEPTH - 1))); do
    echo "[$(date '+%H:%M:%S')] === Evaluating Layer ${layer}/${DEPTH} ==="
    mkdir -p "${RUN_DIR}/readout/l${layer}"
    for task in ${TASKS}; do
        out_json="${RUN_DIR}/readout/l${layer}/${task}.json"
        if [[ -f "${out_json}" ]]; then
            continue
        fi
        if ! .venv/bin/python -m brainmarks_smri vit_mae_3d "${task}" \
            --overrides \
            "imports=[nanobrain.model]" \
            model_kwargs.checkpoint="${CKPT}" \
            model_kwargs.layer="${layer}" \
            output_dir="${RUN_DIR}" \
            name="readout/l${layer}"; then
            echo "FAILED ${task} at layer ${layer}" | tee -a "${RUN_DIR}/readout-failed.txt"
        fi
    done
done

# Evaluate key multi-layer combination
if [[ ${DEPTH} -eq 20 ]]; then
    CAT_LAYERS="15,20"
    CAT_NAME="cat15-20"
elif [[ ${DEPTH} -eq 16 ]]; then
    CAT_LAYERS="13,16"
    CAT_NAME="cat13-16"
else
    CAT_LAYERS="9,12"
    CAT_NAME="cat9-12"
fi

echo "[$(date '+%H:%M:%S')] === Evaluating Concat Readout: ${CAT_NAME} (${CAT_LAYERS}) ==="
mkdir -p "${RUN_DIR}/readout/${CAT_NAME}"
for task in ${TASKS}; do
    out_json="${RUN_DIR}/readout/${CAT_NAME}/${task}.json"
    if [[ -f "${out_json}" ]]; then
        continue
    fi
    if ! .venv/bin/python -m brainmarks_smri vit_mae_3d "${task}" \
        --overrides \
        "imports=[nanobrain.model]" \
        model_kwargs.checkpoint="${CKPT}" \
        model_kwargs.readout_layers="[${CAT_LAYERS}]" \
        output_dir="${RUN_DIR}" \
        name="readout/${CAT_NAME}"; then
        echo "FAILED ${task} at ${CAT_NAME}" | tee -a "${RUN_DIR}/readout-failed.txt"
    fi
done

echo "[$(date '+%H:%M:%S')] Layer sweep finished successfully for ${MODEL_REL_PATH} s${SEED}!"
