#!/usr/bin/env bash
#SBATCH --job-name=layer-sim
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:45:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --no-requeue
#SBATCH --exclude=n-4
#SBATCH --output=/admin/home/dojo/nanobrain/slurms/%x_%j.out
#SBATCH --error=/admin/home/dojo/nanobrain/slurms/%x_%j.err

# Layer similarity (HC paper Fig. 3 and more) for the equal-epoch plain/HC arms, Muon and AdamW.
# usage: sbatch -J nb-l1-similarity scripts/layer_similarity.sbatch [extra args]

set -euo pipefail
cd /admin/home/dojo/nanobrain
set -a
source .env
set +a
export VIRTUAL_ENV=

.venv/bin/python scripts/layer_similarity.py \
    --runs a17/batched-nt l1/dhc4-w384-matched l1/adamw-nt-w384 l1/dhc4-adamw-w384-matched "$@"
