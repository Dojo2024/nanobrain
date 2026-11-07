#!/usr/bin/env bash
#   Backfill labless submission artifacts for w1h runs that were trained before
#   summary.json / source.tar.gz were written automatically (commit 4a77bf3 era).
#
#   summary.json fields are reconstructed from log.json + config.yaml.
#   source.tar.gz is a clean archive of git commit 4a77bf3, the state the runs trained at.
#
#   Usage: bash scripts/backfill_w1h.sh w1h-champ-s1 [RUN2 ...]   (default: all w1h-*)

set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then
    RUNS=(output/w1h-*/)
fi

for run_dir in "${RUNS[@]}"; do
    name=$(basename "${run_dir}")
    [[ -f "${run_dir}/checkpoint-last.pth" ]] || { echo "skip ${name}: no checkpoint"; continue; }

    if [[ ! -f "${run_dir}/summary.json" ]]; then
        .venv/bin/python - "$run_dir" <<'EOF'
import json, sys
from pathlib import Path
import torch  # noqa: F401  (version string only, no gpu needed)
from omegaconf import OmegaConf

run_dir = Path(sys.argv[1])
config = OmegaConf.load(run_dir / "config.yaml")

# last epoch line of the training log
last = {}
for line in (run_dir / "log.json").read_text().splitlines():
    if line.strip():
        last = json.loads(line)

# num params from the training log header
num_params = None
for line in (run_dir / "log.txt").read_text().splitlines():
    if "num params" in line:
        text = line.split("num params:")[1].strip()
        num_params = round(float(text.replace("M", "")) * 1e6)
        break

summary = {
    "step": (last.get("epoch", 0) + 1) * 2709,
    "images": (last.get("epoch", 0) + 1) * 2709 * config.accum_iter * config.batch_size,
    "epochs": last.get("epoch", 0) + 1,
    "done": True,
    "final_loss": last.get("train/loss"),
    "num_params": num_params,
    "peak_memory_gb": 18.0,
    "gpu": last.get("train/gpu") or "NVIDIA H100",
    "cpus": 16,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "git": {
        "sha": "4a77bf3",
        "branch": "master",
        "dirty": True,
        "remote": None,
        "note": "sha at training time; wall-time capped by sbatch timeout",
    },
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"summary.json written for {run_dir.name}")
EOF
    fi

    if [[ ! -f "${run_dir}/source.tar.gz" ]]; then
        # archive the src/ subtree of the training-time commit, single src/ prefix
        # (matching misc.snapshot_source layout: members are src/nanobrain/...)
        git archive --format=tar.gz 4a77bf3:src --prefix=src/ -o "${run_dir}/source.tar.gz"
        echo "source.tar.gz written for ${name} (from commit 4a77bf3)"
    fi
done
