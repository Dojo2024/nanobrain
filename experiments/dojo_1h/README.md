# 1 h ablations of the nanobrain ViT-S MAE

Single-GPU, 1 h wall-clock pretraining runs of the default `ViTMAE3D` (ViT-S, 12 blocks, width 384),
each scored with the frozen nanobrain eval (`python -m nanobrain.eval.run`, 8 default tasks, headline
metric per task). Every arm is run on 2 or 4 seeds (7338, 1, 2, 3) and compared **paired by seed**
against its reference. Runs are from 2026-10-03 and 2026-10-04 on H100 80 GB.

## Result

**Muon + base_lr 3e-3 + a 2-layer readout (layers 9 and 12) beats the original 1 h baseline on 7 of 8
tasks; lesion is flat. It also beats the 40-epoch (~2.5 h) AdamW reference run on every task except lesion.**
The table shows the mean over 4 seeds.

| task | metric | original 1 h<br>(AdamW 1e-3) | AdamW 3e-3 | AdamW 3e-3<br>+ L9,12 | Muon 3e-3 | **Muon 3e-3<br>+ L9,12** | 40-epoch ref<br>(~2.5 h) |
|---|---|---|---|---|---|---|---|
| hippocampus | dice | 0.661 | 0.698 | 0.711 | 0.770 | 0.779 | 0.722 |
| cortex | dice | 0.711 | 0.733 | 0.751 | 0.733 | 0.754 | 0.735 |
| tumor | voxel_auroc | 0.854 | 0.862 | 0.862 | 0.890 | 0.886 | 0.883 |
| lesion | voxel_auroc | 0.721 | 0.748 | 0.739 | 0.724 | 0.729 | 0.743 |
| age | r2 | 0.812 | 0.815 | 0.816 | 0.859 | 0.861 | 0.840 |
| nerve | average_precision | 0.009 | 0.015 | 0.014 | 0.017 | 0.016 | 0.014 |
| vessel | average_precision | 0.008 | 0.010 | 0.013 | 0.013 | 0.013 | 0.012 |
| pmg | auroc | 0.809 | 0.806 | 0.848 | 0.893 | 0.903 | 0.875 |


- **Hippocampus +.118** (.661 → .779), **cortex +.043**, **age +.049**, **tumor +.033**. Every one of these improves on all 4 seeds.
- **Lesion is the one open cost.** It is +.007 against the original, but better on only 1/4 seeds. Muon gives −.024 against AdamW at 3e-3, mostly from one seed (−.065).
- **PMG has a known field-of-view confound,** so treat its numbers with care.
- **Nerve and vessel are near the floor** (average precision ~.01).
- **The 40-epoch reference** is the default config (base_lr 1e-3, 40 epochs, cosine) trained to completion, so ~2.5 h. It is scored with the last-layer readout only.

## What changed, in order

1. **Learning rate.**
   - The default schedule (40 epochs, 5 warmup) is cut by the 1 h budget at epoch ~15, so the learning rate never decays.
   - Fitting the decay into the hour made things worse (`n7/*`): the 1 h model is limited by the learning rate, not by the lack of decay.
   - **base_lr 3e-3 instead of 1e-3, on the default schedule, wins** (`n8/lr3e-3-cut40`).
   - 4e-3 is about the same, 5e-3 is worse, and 8e-3 with a 1-epoch warmup diverges.
2. **Readout (no training).**
   - Single middle layers and the layer mean lose to the last layer.
   - Concatenating layers 9–12 gains on anatomy, but the segmentation probe's L-BFGS does not converge at width 1536, and the eval asserts on that.
   - **Concatenating just layers 9 and 12 (width 768) always converges** and keeps about half the gain: hippocampus +.013 and cortex +.017 on 4/4 seeds, lesion −.009.
   - Adding the class token to the global embedding is null.
3. **Structured masking: negative, retired.**
   - Masking whole groups of patches instead of single patches hurts the dense tasks, and the larger the group, the worse: full axial tubes give hippocampus −.027, 32 mm cubes give −.071.
   - A higher learning rate does not rescue tubes.
4. **Muon: the biggest win.**
   - `torch.optim.Muon` runs on the attention and MLP matrices, and AdamW on everything else, with Muon's update scaled to AdamW's RMS.
   - Tumor and age had not moved with any other change. They track training length in the 40-epoch run, and Muon learns more per step: the epoch-4 loss is .287 against .469 for AdamW.
   - It runs at ~172 vol/s against ~208 for AdamW (torch's Muon steps each matrix in a Python loop), so only 12 epochs fit in the hour, and it still wins.

5. **Doubling Muon's learning rate is negative** (`a17/muon-lr2x`, `muon_lr_mult=2`, AdamW unchanged).
   - It loses on every task: tumor −.025 (0/4), age −.018 (0/4), hippocampus −.045.
   - It learns faster up to epoch ~8, then the loss rises and the grad norm climbs (.16 → .35). The seeds that fit 13 epochs lost the most.
   - 1× stays.

## All arms, paired by seed

The reference is in the `vs` column. Cells are the mean difference (seeds better / seeds), on headline metrics.
`l12` is the default readout (last block). `catA-B` concatenates the normed output of blocks A and B.

| arm | vs | n | hippocampus | cortex | tumor | lesion | age | nerve | vessel | pmg |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` l3 | `baseline` l12 | 4 | -0.082 (0/4) | -0.024 (0/4) | -0.041 (0/4) | -0.046 (0/4) | -0.044 (0/4) | -0.003 (0/4) | -0.002 (0/4) | -0.052 (0/4) |
| `baseline` l6 | `baseline` l12 | 4 | -0.051 (0/4) | -0.008 (0/4) | -0.026 (0/4) | -0.026 (0/4) | -0.009 (1/4) | -0.002 (0/4) | -0.002 (0/4) | +0.020 (3/4) |
| `baseline` l9 | `baseline` l12 | 4 | -0.020 (0/4) | -0.002 (0/4) | -0.012 (0/4) | -0.015 (1/4) | -0.002 (1/4) | -0.001 (0/4) | -0.001 (0/4) | +0.014 (2/4) |
| `baseline` mean3-6-9-12 | `baseline` l12 | 4 | -0.029 (0/4) | +0.001 (4/4) | -0.011 (0/4) | -0.018 (0/4) | -0.002 (2/4) | -0.001 (0/4) | -0.001 (0/4) | +0.007 (3/4) |
| `baseline` cat9-10-11-12 | `baseline` l12 | 4 | +0.027 (3/3) | +0.030 (1/1) | -0.003 (1/4) | -0.003 (1/4) | +0.002 (2/4) | -0.001 (1/3) | +0.001 (2/3) | +0.039 (3/3) |
| `n7/cos14` l12 | `baseline` l12 | 4 | -0.038 (0/4) | -0.006 (1/4) | -0.010 (0/4) | -0.036 (0/4) | -0.016 (0/4) | -0.002 (0/4) | -0.001 (0/4) | +0.080 (4/4) |
| `n7/lin14` l12 | `baseline` l12 | 2 | -0.019 (0/2) | -0.004 (0/2) | -0.004 (0/2) | -0.037 (0/2) | -0.010 (0/2) | -0.002 (1/2) | -0.001 (0/2) | +0.095 (2/2) |
| `n7/time` l12 | `baseline` l12 | 2 | -0.025 (0/2) | -0.004 (0/2) | -0.004 (1/2) | -0.019 (1/2) | -0.016 (0/2) | -0.002 (0/2) | -0.002 (0/2) | +0.089 (2/2) |
| `n8/lr2e-3` l12 | `baseline` l12 | 2 | +0.018 (2/2) | +0.013 (2/2) | +0.008 (2/2) | -0.003 (1/2) | +0.004 (2/2) | +0.001 (1/2) | -0.001 (0/2) | +0.049 (2/2) |
| `n8/lr3e-3` l12 | `baseline` l12 | 4 | +0.010 (2/4) | +0.016 (4/4) | +0.006 (3/4) | +0.002 (2/4) | +0.003 (3/4) | -0.000 (2/4) | -0.002 (0/4) | +0.043 (3/4) |
| `n8/lr5e-3` l12 | `baseline` l12 | 2 | -0.092 (0/2) | -0.026 (0/2) | -0.028 (0/2) | -0.060 (0/2) | -0.051 (0/2) | -0.005 (0/2) | -0.004 (0/2) | -0.007 (1/2) |
| `n8/lr8e-3` l12 | `baseline` l12 | 2 | -0.537 (0/2) | -0.346 (0/1) | -0.223 (0/2) | -0.129 (0/2) | -0.340 (0/2) | -0.008 (0/2) | -0.007 (0/2) | -0.095 (0/2) |
| `n8/lr3e-3-cut40` l12 | `baseline` l12 | 4 | +0.037 (4/4) | +0.023 (4/4) | +0.008 (2/4) | +0.027 (3/4) | +0.002 (2/4) | +0.006 (4/4) | +0.003 (4/4) | -0.003 (2/4) |
| `n8/lr4e-3-cut40` l12 | `n8/lr3e-3-cut40` l12 | 1 | +0.011 (1/1) | -0.002 (0/1) | -0.010 (0/1) | +0.017 (1/1) | +0.002 (1/1) | -0.000 (0/1) | -0.000 (0/1) | +0.054 (1/1) |
| `n8/lr5e-3-cut40` l12 | `n8/lr3e-3-cut40` l12 | 2 | -0.015 (0/2) | -0.002 (0/2) | -0.027 (0/2) | +0.011 (2/2) | +0.002 (1/2) | -0.002 (0/2) | -0.003 (0/2) | -0.016 (1/2) |
| `b2/tube-si` l12 | `n8/lr3e-3-cut40` l12 | 2 | -0.016 (0/2) | -0.009 (0/2) | -0.017 (0/2) | +0.022 (2/2) | -0.001 (1/2) | -0.003 (0/2) | -0.001 (0/2) | +0.040 (1/2) |
| `b2/tube-si-lr5e-3` l12 | `n8/lr3e-3-cut40` l12 | 4 | -0.027 (0/4) | -0.011 (0/4) | -0.008 (1/4) | +0.018 (3/4) | -0.002 (3/4) | -0.004 (0/4) | -0.002 (0/4) | +0.034 (3/4) |
| `b1/block4` l12 | `n8/lr3e-3-cut40` l12 | 4 | -0.071 (0/4) | -0.041 (0/4) | -0.028 (0/4) | -0.027 (0/4) | -0.003 (2/4) | -0.008 (0/4) | -0.004 (0/4) | -0.032 (1/4) |
| `n8/lr3e-3-cut40` cat6-12 | `n8/lr3e-3-cut40` l12 | 4 | +0.011 (4/4) | +0.021 (4/4) | -0.004 (1/4) | -0.012 (0/4) | -0.001 (2/4) | -0.002 (0/4) | +0.002 (4/4) | +0.066 (4/4) |
| `n8/lr3e-3-cut40` cat9-12 | `n8/lr3e-3-cut40` l12 | 4 | +0.013 (4/4) | +0.017 (4/4) | +0.001 (2/4) | -0.009 (1/4) | +0.002 (2/4) | -0.002 (0/4) | +0.002 (4/4) | +0.042 (4/4) |
| `n8/lr3e-3-cut40` cls-l12 | `n8/lr3e-3-cut40` l12 | 4 | – | – | – | – | +0.000 (3/4) | – | – | +0.036 (3/4) |
| `a17/muon` l12 | `n8/lr3e-3-cut40` l12 | 4 | +0.072 (4/4) | -0.001 (2/4) | +0.028 (4/4) | -0.024 (1/4) | +0.044 (4/4) | +0.002 (4/4) | +0.002 (4/4) | +0.087 (4/4) |
| `a17/muon` cat9-12 | `n8/lr3e-3-cut40` cat9-12 | 4 | +0.068 (4/4) | +0.003 (3/4) | +0.024 (4/4) | -0.011 (2/4) | +0.045 (4/4) | +0.003 (4/4) | -0.000 (2/4) | +0.055 (3/4) |
| `a17/muon-lr2x` l12 | `a17/muon` l12 | 4 | -0.045 (1/4) | -0.006 (1/4) | -0.025 (0/4) | -0.007 (2/4) | -0.018 (0/4) | -0.004 (1/4) | -0.001 (1/4) | -0.022 (1/4) |
| `a17/muon-lr2x` cat9-12 | `a17/muon` cat9-12 | 4 | -0.042 (1/4) | -0.006 (1/4) | -0.032 (0/4) | -0.009 (1/4) | -0.028 (0/4) | -0.003 (1/4) | -0.001 (1/4) | -0.004 (2/4) |

## Code

- **`optimizer: muon`, `muon_lr_mult`** (`train.py`, `utils/misc.py: MuonAdamW`).
  - Muon covers the weight matrices whose names start with `misc.MUON_PREFIXES`; AdamW covers the rest.
  - Muon uses `adjust_lr_fn="match_rms_adamw"`, so the existing `base_lr`, schedule and `weight_decay` carry over unchanged.
  - With the default `optimizer: adamw`, behaviour is unchanged.
- **`mask_groups`** (`model.py: make_mask_groups`, `sample_sequences`). These are structured masking group shapes in patches. `null` (the default) is bit-identical to the original sampler.
- **`lr_decay: cosine|linear` and `time_budget`** (`train.py`, `utils/misc.py: WarmupThenLinear`). `time_budget` keys the schedule to wall time.
- **`src/nanobrain/readout.py`.** `ReadoutViTMAE3D` subclasses `ViTMAE3D` and reads out a concatenation or mean of several blocks. It adds no weights.
  - `python -m nanobrain.readout CKPT OUT --layers 9 12` writes a checkpoint the unmodified eval loads.
  - With `--layers 12` it is bit-exact with `ViTMAE3D`.
- **`scripts/arm_1hr.sbatch`.** One seed of one arm: 1 h `timeout` training, then the eval on the last layer and on the layers 9+12 readout.
  - `scripts/readout_eval.sbatch` runs readout-only sweeps on an existing run.
  - The SLURM partition and account in the headers are specific to our cluster.

Reproduce one seed of the best recipe:

```bash
sbatch -J a17-muon-s7338 scripts/arm_1hr.sbatch a17/muon 7338 base_lr=0.003 optimizer=muon
```

## Files here

- `results.csv`: one row per run, seed, readout and task, with the headline metric and its 95% CI.
  - Only scalar metrics are published. The eval JSONs also hold per-subject targets and predictions, which are left out.
- `runs.csv`: one row per run and seed, with the config overrides against `default_pretrain.yaml`, epochs completed in the hour, final train loss and mean throughput.
- `collect_results.py`: rebuilds both CSVs from a run output directory.

## Open questions

- **Which budget counts:** wall clock or FLOPs? Muon costs ~17% throughput. A batched Newton–Schulz step would recover most of that, which matters only under a wall-clock rule.
- **Is a readout defined in the model class a fair submission?** Examples are `dense_embed`/`global_embed` over several blocks. The eval code itself is untouched.
