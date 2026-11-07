#!/usr/bin/env bash
#
#   1-hour winners wave (Labless-1h protocol): our validated 1E recipes,
#   re-run at 60 min + new-harness eval. 4 seeds each.
#
#   Usage: sbatch wave1h.sbatch <ARM> <SEED>
#     ARM in: champ | segcombo | w1152 | champ-muonx2
#
#SBATCH --job-name=nb-v3-w1h
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=2:30:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --output=slurms/w1h_%j.out
#SBATCH --error=slurms/w1h_%j.err

set -euo pipefail

cd /admin/home/dojo/nanobrain-v3

export BRAINMARKS_SMRI_DATA=/data/brainmarks-smri
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM="$1"
SEED="$2"
NAME="w1h-${ARM}-s${SEED}"
RUN_DIR="output/${NAME}"
mkdir -p "${RUN_DIR}"

# shared scaffold: the 1E champion recipe
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

# extra per-arm overrides (placed AFTER COMMON, so these win on conflicts)
case "${ARM}" in
champ)        EXTRA=( epochs=15 warmup_epochs=2 base_lr=0.0035 ) ;;
segcombo)     EXTRA=( epochs=17 warmup_epochs=2 base_lr=0.003
                  model_kwargs.decoder_depth=2 model_kwargs.decoder_embed_dim=384 model_kwargs.decoder_num_heads=6
                  cooldown_epochs=1 cooldown_num_visible=1280 ) ;;
w1152)        EXTRA=( epochs=12 warmup_epochs=2 base_lr=0.0035 model_kwargs.embed_dim=1152 ) ;;
champ-muonx2) EXTRA=( epochs=15 warmup_epochs=2 base_lr=0.0035 muon_lr_mult=2.0 ) ;;
speedchamp)   EXTRA=( epochs=15 warmup_epochs=2 base_lr=0.0032
                  batch_size=64 accum_iter=1 num_samples=4 num_predict=1280 ) ;;
anneal)       EXTRA=( epochs=11 warmup_epochs=2 base_lr=0.0035 ) ;;
rampdown)     EXTRA=( epochs=8 warmup_epochs=2 base_lr=0.0035
                  visible_ramp_start=1024 visible_ramp_end=384 ) ;;
champ2h)      EXTRA=( epochs=23 warmup_epochs=2 base_lr=0.0035 ); TRAIN_CAP="120m" ;;
muonx2h)      EXTRA=( epochs=23 warmup_epochs=2 base_lr=0.0035 muon_lr_mult=2.0 ); TRAIN_CAP="120m" ;;
# --- next screening wave (2 seeds each) ---
# leader's data recipe (qc-filtered filelist) on our model; clip_quantiles already our default
champqc)      EXTRA=( epochs=15 warmup_epochs=2 base_lr=0.0035
                  train_filelist=src/nanobrain/config/fomo300k_images_qc.txt ) ;;
# width axis with a schedule that completes inside the hour (measured ~0.12 s/it, <~10 epochs fit)
w1152a)       EXTRA=( epochs=9 warmup_epochs=2 base_lr=0.0035 model_kwargs.embed_dim=1152 ) ;;
# opposite curriculum: harder task first
rampup)       EXTRA=( epochs=8 warmup_epochs=2 base_lr=0.0035
                  visible_ramp_start=384 visible_ramp_end=1024 ) ;;
# one extra sample sequence per image (richer steps at 1.5x cost; bs64/ns4 OOM'd)
ns3)          EXTRA=( epochs=8 warmup_epochs=2 base_lr=0.0035 num_samples=3 ) ;;
# muon x2 tracked better than champ at ep4 (0.70 vs 0.72); give it a completing schedule too
muonx2a)      EXTRA=( epochs=11 warmup_epochs=2 base_lr=0.0035 muon_lr_mult=2.0 ) ;;
# --- seg-focused screening ---
# dense decode: predict 2x the patches - denser reconstruction signal for the seg probes;
# only the decoder grows (~1.6x), est ~0.125 s/it so 9 annealing epochs fit the hour
champdense)   EXTRA=( epochs=9 warmup_epochs=2 base_lr=0.0035 num_predict=1280 ) ;;
# foreground re-weight sweep: seg probes read tissue features
fg05)         EXTRA=( epochs=11 warmup_epochs=2 base_lr=0.0035 model_kwargs.fg_loss_weight=0.5 ) ;;
*)            echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

# training cap, in minutes (timeout exits 124 when it stops training: expected)
TRAIN_CAP="${TRAIN_CAP:-60m}"
status=0
timeout --kill-after=2m "${TRAIN_CAP}" \
    .venv/bin/python -m nanobrain.train \
    --overrides name="${NAME}" output_dir=output "${COMMON[@]}" "${EXTRA[@]}" \
    || status=$?
if [[ $status -ne 0 && $status -ne 124 ]]; then
    echo "training failed with ${status}" >&2
    exit ${status}
fi

# eval with the current v2 harness (open = the 11 labless-scored tasks), final layer only
uv run --no-sync python -m brainmarks_smri vit_mae_3d open \
    --overrides \
    "imports=[nanobrain.model]" \
    model_kwargs.checkpoint="${RUN_DIR}/checkpoint-last.pth" \
    model_kwargs.readout_layers=null \
    output_dir="${RUN_DIR}" \
    name=eval

echo "=== wave1h ${ARM} s${SEED} done (train status ${status}) ==="
