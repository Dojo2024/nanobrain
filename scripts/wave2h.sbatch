#!/usr/bin/env bash
#
#   Wave 2: 8 arms x 4 seeds (7338, 1, 2, 3) = 32 x 1h runs on the v3 champion
#   scaffold. Stacks the axes that measured positive: muon-x2 (+0.002 over champ),
#   width 1152 (+0.002 via w1152-s1 0.5300), anneal (+0.001), segcombo (+0.001).
#   Same protocol as wave1h: epoch-based fitting with a 60m train cap, v2-harness eval.
#
#   Usage: sbatch wave2h.sbatch <ARM> <SEED>
#SBATCH --job-name=nb-v3-w2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2:30:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --output=slurms/w2_%j.out
#SBATCH --error=slurms/w2_%j.err

set -euo pipefail

cd /admin/home/dojo/nanobrain-v3

export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM="$1"
SEED="$2"
NAME="w2-${ARM}-s${SEED}"
RUN_DIR="output/${NAME}"
mkdir -p "${RUN_DIR}"

# shared scaffold: the 1E champion recipe (identical to wave1h)
COMMON=(
    seed="${SEED}"
    batch_size=32 accum_iter=1
    optimizer=muon muon_impl=fused
    weight_decay=0.05
    num_visible=384 num_predict=640 num_samples=2
    model_kwargs.embed_dim=1024 model_kwargs.depth=8 model_kwargs.num_heads=16
    model_kwargs.decoder_depth=4 model_kwargs.decoder_embed_dim=512 model_kwargs.decoder_num_heads=8
    model_kwargs.class_tokens=1 model_kwargs.drop_path_rate=0.0
    model_kwargs.norm_target=true
    model_kwargs.grad_loss_weight=0.5
    model_kwargs.fg_loss_weight=0.25
    model_kwargs.dense_neighbor_pool=true
    wandb=false
)

case "${ARM}" in
mx2-w1152)     EXTRA=( epochs=9  warmup_epochs=2 base_lr=0.0035 muon_lr_mult=2.0 model_kwargs.embed_dim=1152 ) ;;
mx2-w1152an)   EXTRA=( epochs=8  warmup_epochs=2 base_lr=0.0035 muon_lr_mult=2.0 model_kwargs.embed_dim=1152 ) ;;
mx2-w1152seg)  EXTRA=( epochs=8  warmup_epochs=2 base_lr=0.0035 muon_lr_mult=2.0 model_kwargs.embed_dim=1152
                 model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6
                 cooldown_epochs=1 cooldown_num_visible=1280 ) ;;
mx2-anseg)     EXTRA=( epochs=13 warmup_epochs=2 base_lr=0.0035 muon_lr_mult=2.0
                 model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6
                 cooldown_epochs=1 cooldown_num_visible=1280 ) ;;
full)          EXTRA=( epochs=8  warmup_epochs=2 base_lr=0.0035 muon_lr_mult=2.0 model_kwargs.embed_dim=1152
                 model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6
                 cooldown_epochs=1 cooldown_num_visible=1280 ) ;;
mx2-w1152cd)   EXTRA=( epochs=7  warmup_epochs=2 base_lr=0.0035 muon_lr_mult=2.0 model_kwargs.embed_dim=1152 num_predict=1280 ) ;;
mx2-w1152fg)   EXTRA=( epochs=9  warmup_epochs=2 base_lr=0.0035 muon_lr_mult=2.0 model_kwargs.embed_dim=1152
                 model_kwargs.fg_loss_weight=0.5 ) ;;
mx2a)          EXTRA=( epochs=11 warmup_epochs=2 base_lr=0.0035 muon_lr_mult=2.0 ) ;;
*)             echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

status=0
timeout --kill-after=2m "60m" \
    .venv/bin/python -m nanobrain.train \
    --overrides name="${NAME}" output_dir=output "${COMMON[@]}" "${EXTRA[@]}" \
    || status=$?
if [[ $status -ne 0 && $status -ne 124 ]]; then
    echo "training failed with ${status}" >&2
    exit ${status}
fi

uv run --no-sync python -m brainmarks_smri vit_mae_3d open \
    --overrides \
    "imports=[nanobrain.model]" \
    model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
    model_kwargs.readout_layers=null \
    output_dir="${RUN_DIR}" \
    name=eval

echo "=== wave2 ${ARM} s${SEED} done (train status ${status}) ==="
