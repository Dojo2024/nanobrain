"""Generate the iso-flop 1h sweep configs for the original nanobrain.

Measures each config's flops/step with FlopCounterMode (cpu, same counter the trainer
logs; measured at batch 1 / 1 sample and scaled - flops are exactly linear in both),
then sets total_steps = budget / flops so every config trains the same compute. Also
sets time_limit per config from the measured pace.

Axes: width x depth on the iso-flop diagonal, mlp ratio, attention heads, and the
masking ratio (num_visible at fixed num_predict). The visible-ramp (masking-ratio
curriculum) arms come later, in combine_winners.py, on the winner.

Usage: uv run python scripts/gen_isoflop_sweep.py [--budget 7.85e17]
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nanobrain.model import ViTMAE3D  # noqa: E402
from nanobrain.train import count_batch_flops  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "src/nanobrain/config/default_pretrain.yaml"

# (name, embed_dim, depth, num_heads, mlp_ratio, num_visible)
# heads always divide the width; the masking-ratio axis varies num_visible at np 1920
GRID = [
    ("base-e768-d12",       768,  12, 12, 4, 640),
    ("wide-e640-d16-h10",   640,  16, 10, 4, 640),
    ("wide-e896-d10-h14",   896,  10, 14, 4, 640),
    ("wide-e1024-d8-h16",  1024,   8, 16, 4, 640),
    ("wide-e1152-d6-h12",  1152,   6, 12, 4, 640),
    ("mlp6-e768-d10",       768,  10, 12, 6, 640),
    ("heads8-e768-d12",     768,  12,  8, 4, 640),
    ("heads16-e768-d12",    768,  12, 16, 4, 640),
    ("mask-nv384",          768,  12, 12, 4, 384),
    ("mask-nv896",          768,  12, 12, 4, 896),
    ("mask-nv1152",         768,  12, 12, 4, 1152),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=7.85e17, help="flops budget per run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[7338, 1])
    ap.add_argument("--tflops-achieved", type=float, default=205.0,
                    help="measured sustained tflop/s for wall estimates")
    args = ap.parse_args()

    base_cfg = OmegaConf.load(BASE)
    outdir = ROOT / "configs" / "iso"
    outdir.mkdir(parents=True, exist_ok=True)
    plan = []
    for name, width, depth, heads, mlp, nv in GRID:
        cfg = copy.deepcopy(base_cfg)
        cfg["name"] = name
        cfg["num_visible"] = nv
        cfg["model_kwargs"].update(embed_dim=width, depth=depth, num_heads=heads, mlp_ratio=mlp)
        model = ViTMAE3D.from_config(cfg)
        probe = copy.deepcopy(cfg)
        probe.batch_size, probe.num_samples = 1, 1  # count small, scale exactly
        bflops = count_batch_flops(probe, model, torch.device("cpu")) / 1e12
        bflops *= cfg.batch_size * cfg.num_samples
        del model
        steps = round(args.budget / (bflops * 1e12))
        wall_min = steps * bflops / args.tflops_achieved / 60
        cfg["total_steps"] = steps
        cfg["time_limit"] = min(90, int(wall_min) + 8)
        OmegaConf.save(cfg, outdir / f"{name}.yaml")
        plan.append((name, round(bflops, 2), steps, round(wall_min, 1)))

    w = max(len(p[0]) for p in plan)
    print(f"{'config':{w}}  TFLOP/step  steps   est wall (min)")
    for name, bf, steps, wall in plan:
        print(f"{name:{w}}  {bf:10.2f}  {steps:6d}  {wall:8.1f}")
    print(f"\nbudget {args.budget:.2e} flops/run; launch ({len(plan) * len(args.seeds)} jobs):")
    for name, *_ in plan:
        for s in args.seeds:
            print(f"sbatch --time=2:45:00 scripts/sweep_run.sbatch iso/{name} {s}")


if __name__ == "__main__":
    main()
