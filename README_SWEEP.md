# nanobrain-sweep — iso-flop architecture sweep on the original nanobrain

Fresh clone of the original nanobrain at `440fc92`, branch `labless/iso-sweep`.
**Nothing is submitted and no job runs** — every script prints its plan and stops.

## What changed vs upstream (the diff, all of it minimal)

| Change | Files | Size |
|---|---|---|
| visible-ramp port (progressive masking; the `num_visible` interpolation) | `train.py`, `config/default_pretrain.yaml` | the one code feature: +6 lines |
| iso-flop config generator — flops/step measured per config, `total_steps` set to the budget | `scripts/gen_isoflop_sweep.py` | new script |
| generic sweep runner (per config + seed; step-based, `time_limit` from its config) | `scripts/sweep_run.sbatch` | new script |
| sweep submit wrapper — prints the sbatch plan; sbatch lines commented | `scripts/submit_sweep.sh` | new script |
| winner combination / ablation generator — per-axis winners, combo configs, parked | `scripts/combine_winners.py` | new script |
| README for the sweep protocol | `README_SWEEP.md` | new doc |

Everything else is stock upstream: step-based training (`total_steps`), native
`time_limit`, flops accounting (`train_flops` in summary.json), SIGTERM-safe
summaries with auto-resume, the labless submit script with git provenance
(`origin remote in git info`), and the eval harness.

## The grid

8 configs × 2 seeds = 16 runs ≈ 28 GPU-hours, all at the same ~7.85e17-flops
budget class. Width × depth on the iso-flop diagonal, plus mlp-ratio and
head-count around the base.

| axis | candidates | winner (so far) |
|---|---|---|
| width | 640 / 768 / 896 / 1024 / 1152 | ? |
| depth (at matched flops) | 6 / 8 / 10 / 12 / 16 | ? |
| mlp ratio | 4 / 6 | ? |
| heads | 8 / 12 / 16 | ? |
| + ramp arms | on / off | ? |

## Protocol

Same as every other run on the board: step-based, 1h-class budget, muon ×2,
the original 1mm grid, qc-target file, summaries that survive SIGTERM and
auto-resume on relaunch, flops + provenance recorded per run. **Not submitted**;
entirely optional to run at all — it prepares rather than executes.
