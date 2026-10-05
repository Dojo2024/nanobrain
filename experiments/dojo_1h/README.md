# 1 h ablations of the nanobrain ViT-S MAE

Single-GPU, 1 h **wall-clock** pretraining runs of the default `ViTMAE3D` (ViT-S, 12 blocks, width 384),
each scored with the frozen nanobrain eval (`python -m nanobrain.eval.run`, 8 default tasks, headline
metric per task). Arms run on 2–4 seeds (7338, 1, 2, 3) and are compared **paired by seed** against their
reference. Runs are from 2026-10-03 to 2026-10-05 on H100 80 GB.

## Result

**Best config: Muon (batched) + base_lr 3e-3 + normalized targets, read out from layers 9 and 12.**

| task | metric | original 1 h<br>(AdamW 1e-3) | **best** | change | 40-epoch AdamW ref<br>(~2.5 h, 1 seed) |
|---|---|---|---|---|---|
| hippocampus | dice | .653 | **.801** | +.148 | .722 |
| cortex | dice | .710 | **.765** | +.055 | .735 |
| tumor | voxel AUROC | .854 | **.898** | +.044 | .883 |
| age | R² | .812 | **.879** | +.067 | .840 |
| nerve | AP | .009 | .021 | +.011 | .014 |
| vessel | AP | .007 | .015 | +.008 | .012 |
| pmg | AUROC | .808 | .912 | +.104 | .875 |
| lesion | voxel AUROC | .716 | .707 | −.010 | .743 |

- **Every task except lesion improves, and all of those beat the ~2.5 h reference.** Both 1 h columns are the mean over the same 3 clean seeds (7338, 1, 2); best is `a17/batched-nt` with cat9-12. The 4-seed run with torch's Muon (`a3/normtarget`, cat9-12) is within ±.004 of it.
- **Lesion is the open problem.** Every change that helps the structure tasks has cost it. Its best was .748, with AdamW at 3e-3 and the last-layer readout.
- **PMG has a known field-of-view confound,** so treat its numbers with care.
- **Nerve and vessel are near the floor** (average precision ~.01–.02).

Reproduce one seed:

```bash
sbatch -J best-s7338 scripts/arm_1hr.sbatch best 7338 base_lr=0.003 optimizer=muon muon_impl=batched norm_target=true
```

The script trains for 1 h, then scores the last layer (`eval/`) and the layers 9+12 readout (`readout/cat9-12/`).

## What we learned, in order

1. **The learning rate was too low.**
   - The default 40-epoch cosine schedule is cut at epoch ~15 by the 1 h budget, so it never decays.
   - Fitting the decay into the hour made things worse (`n7/*`): the 1 h model is limited by the learning rate, not by the lack of decay.
   - **base_lr 3e-3 instead of 1e-3, on the default schedule**, gave hippocampus +.037. 5e-3 is worse, and 8e-3 diverges.
2. **A 2-layer readout.**
   - Single middle layers and the layer mean lose to the last layer.
   - Concatenating layers 9–12 gains on anatomy, but the probe's L-BFGS does not converge at width 1536, and the eval asserts on that.
   - **Layers 9 + 12 (width 768) always converge** and keep about half the gain. The class token in the global embedding is null.
3. **Structured masking is negative.** Tubes or 32 mm cubes instead of random patches hurt the dense tasks, and the larger the hidden region, the worse.
4. **Muon, the biggest single win.**
   - `torch.optim.Muon` runs on the attention and MLP matrices, and AdamW on the rest. Muon's update is scaled to AdamW's RMS, so the lr schedule carries over.
   - Hippocampus +.072, age +.044 and tumor +.028 vs AdamW 3e-3. Tumor and age had not moved with any other change.
   - Doubling Muon's lr is negative: the run goes unstable after epoch ~8.
5. **Balanced sampling by scan type is null.**
   - Pretraining is 66% T1w and has no ADC.
   - Square-root sampling (T1 42%, DWI 29%) didn't move lesion or tumor, and the T1 tasks barely noticed the T1 cut.
6. **EMA of the weights is null.** Averages over 1k and 2k steps, scored within the same run, changed every task by ≤.005.
7. **Normalized targets (MAE's norm_pix_loss) win.**
   - Each target patch is standardized by its in-mask mean and variance.
   - Hippocampus +.021, cortex +.009, tumor, age and PMG up vs Muon; lesion −.009.
8. **Batched Muon is the same update, 16% faster.**
   - Same-shape matrices are orthogonalized in one batched Newton–Schulz. It runs at 205 vs 178 vol/s, so 15 vs 13 epochs in the hour.
   - Scores are within noise, so the model isn't step-limited any more. Adopted for the free speed.
9. **A brightness head is mixed.**
   - With normalized targets, the decoder also predicts each patch's intensity mean and log std.
   - Lesion +.021, but mostly one seed; age −.016 (0/3). Not adopted.

## Measurement notes

- **Same-seed reruns are not reproducible.** Two runs with identical config and seeds differed by up to .024 on tumor and .013 on lesion. The causes are GPU nondeterminism, and the 1 h wall clock fitting 12 or 13 epochs depending on node speed. So paired diffs of ~.01 on tumor, lesion or PMG are noise. Within-run comparisons (readouts, EMA) are immune to this.
- **Slow GPU episodes.** Four times, one job's GPU ran at ~0.42 s/step instead of ~0.16 while other jobs on the same node were normal. The affected seed gets fewer epochs and is reported separately in the table below.

## All runs vs the original 1 h baseline

Cells are mean score (paired diff vs the original baseline on the same seeds). Rows with fewer seeds are paired only
on those seeds, so their absolute means aren't directly comparable row to row. Each verdict is against that arm's own
reference: masking vs AdamW 3e-3, later arms vs Muon.

| model | seeds | epochs | hippo | cortex | tumor | lesion | age | nerve | vessel | pmg | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Original 1 h baseline (AdamW 1e-3) | 4 | 14/15 | **0.661** | **0.711** | **0.854** | **0.721** | **0.812** | **0.009** | **0.008** | **0.809** | reference |
| **Schedule (N7)** | | | | | | | | | | | |
| decay fit to 1 h, cosine | 4 | 14 | 0.623 (-0.038) | 0.705 (-0.006) | 0.844 (-0.010) | 0.685 (-0.036) | 0.796 (-0.016) | 0.007 (-0.002) | 0.006 (-0.001) | 0.889 (+0.080) | negative |
| decay fit to 1 h, linear | 2 | 14 | 0.631 (-0.019) | 0.705 (-0.004) | 0.850 (-0.004) | 0.679 (-0.037) | 0.802 (-0.010) | 0.007 (-0.002) | 0.006 (-0.001) | 0.888 (+0.095) | negative |
| schedule keyed to wall time | 2 | 15 | 0.625 (-0.025) | 0.706 (-0.004) | 0.850 (-0.004) | 0.696 (-0.019) | 0.796 (-0.016) | 0.007 (-0.002) | 0.006 (-0.002) | 0.882 (+0.089) | negative |
| **Learning rate (N8)** | | | | | | | | | | | |
| 2e-3, 1 h decay | 2 | 14 | 0.668 (+0.018) | 0.722 (+0.013) | 0.862 (+0.008) | 0.712 (-0.003) | 0.816 (+0.004) | 0.009 (+0.001) | 0.006 (-0.001) | 0.843 (+0.049) | positive |
| 3e-3, 1 h decay | 4 | 14 | 0.671 (+0.010) | 0.727 (+0.016) | 0.860 (+0.006) | 0.723 (+0.002) | 0.816 (+0.003) | 0.009 (-0.000) | 0.006 (-0.002) | 0.852 (+0.043) | positive |
| 5e-3, 1 h decay | 2 | 14 | 0.558 (-0.092) | 0.684 (-0.026) | 0.826 (-0.028) | 0.656 (-0.060) | 0.762 (-0.051) | 0.004 (-0.005) | 0.003 (-0.004) | 0.786 (-0.007) | unstable |
| 8e-3, 1 h decay | 2 | 14 | 0.113 (-0.537) | 0.365 (-0.346) | 0.631 (-0.223) | 0.586 (-0.129) | 0.472 (-0.340) | 0.000 (-0.008) | 0.001 (-0.007) | 0.698 (-0.095) | diverged |
| **3e-3, default schedule** | 4 | 15 | 0.698 (+0.037) | 0.733 (+0.023) | 0.862 (+0.008) | 0.748 (+0.027) | 0.815 (+0.002) | 0.015 (+0.006) | 0.010 (+0.003) | 0.806 (-0.003) | **adopted** |
| 4e-3, default schedule | 1 | 15 | 0.720 (+0.075) | 0.735 (+0.028) | 0.874 (+0.027) | 0.754 (+0.011) | 0.835 (+0.016) | 0.017 (+0.009) | 0.012 (+0.005) | 0.920 (+0.090) | ≈ 3e-3 (1 seed) |
| 5e-3, default schedule | 2 | 15 | 0.683 (+0.034) | 0.731 (+0.022) | 0.841 (-0.013) | 0.750 (+0.035) | 0.820 (+0.008) | 0.013 (+0.004) | 0.008 (+0.000) | 0.808 (+0.015) | worse |
| **Readout on original baseline** | | | | | | | | | | | |
| layer 3 | 4 | 14/15 | 0.579 (-0.082) | 0.686 (-0.024) | 0.813 (-0.041) | 0.675 (-0.046) | 0.769 (-0.044) | 0.006 (-0.003) | 0.005 (-0.002) | 0.757 (-0.052) | negative |
| layer 6 | 4 | 14/15 | 0.610 (-0.051) | 0.703 (-0.008) | 0.828 (-0.026) | 0.696 (-0.026) | 0.803 (-0.009) | 0.008 (-0.002) | 0.006 (-0.002) | 0.829 (+0.020) | negative |
| layer 9 | 4 | 14/15 | 0.641 (-0.020) | 0.709 (-0.002) | 0.842 (-0.012) | 0.706 (-0.015) | 0.810 (-0.002) | 0.009 (-0.001) | 0.007 (-0.001) | 0.823 (+0.014) | negative |
| mean of 3, 6, 9, 12 | 4 | 14/15 | 0.632 (-0.029) | 0.712 (+0.001) | 0.843 (-0.011) | 0.703 (-0.018) | 0.810 (-0.002) | 0.009 (-0.001) | 0.006 (-0.001) | 0.816 (+0.007) | negative |
| concat 9–12 (4 layers) | 4 | 14/15 | 0.693 (+0.027) | 0.744 (+0.030) | 0.851 (-0.003) | 0.719 (-0.003) | 0.814 (+0.002) | 0.009 (-0.001) | 0.009 (+0.001) | 0.841 (+0.039) | probe fails to converge |
| **Readout on AdamW 3e-3** | | | | | | | | | | | |
| concat 6+12 | 4 | 15 | 0.709 (+0.048) | 0.754 (+0.043) | 0.858 (+0.004) | 0.737 (+0.015) | 0.814 (+0.001) | 0.013 (+0.004) | 0.012 (+0.004) | 0.872 (+0.063) | positive |
| **concat 9+12** | 4 | 15 | 0.711 (+0.050) | 0.751 (+0.040) | 0.862 (+0.008) | 0.739 (+0.018) | 0.816 (+0.004) | 0.014 (+0.004) | 0.013 (+0.005) | 0.848 (+0.039) | **adopted** |
| class token in global | 4 | 15 | – | – | – | – | 0.815 (+0.002) | – | – | 0.842 (+0.033) | null |
| **Masking (B1/B2), AdamW 3e-3** | | | | | | | | | | | |
| axial tubes | 2 | 15 | 0.682 (+0.033) | 0.724 (+0.015) | 0.851 (-0.004) | 0.761 (+0.046) | 0.817 (+0.005) | 0.013 (+0.004) | 0.010 (+0.002) | 0.865 (+0.071) | negative |
| axial tubes, lr 5e-3 | 4 | 15 | 0.671 (+0.010) | 0.723 (+0.012) | 0.854 (+0.000) | 0.767 (+0.045) | 0.813 (+0.001) | 0.012 (+0.002) | 0.009 (+0.001) | 0.840 (+0.031) | negative |
| 32 mm blocks | 4 | 15 | 0.627 (-0.034) | 0.692 (-0.018) | 0.834 (-0.020) | 0.721 (-0.000) | 0.812 (-0.000) | 0.007 (-0.002) | 0.007 (-0.001) | 0.774 (-0.035) | negative |
| **Muon (A17), lr 3e-3** | | | | | | | | | | | |
| **Muon** | 4 | 12 | 0.770 (+0.109) | 0.733 (+0.022) | 0.890 (+0.036) | 0.724 (+0.003) | 0.859 (+0.047) | 0.017 (+0.008) | 0.013 (+0.005) | 0.893 (+0.084) | **adopted** |
| Muon + concat 9+12 | 4 | 12 | 0.779 (+0.118) | 0.754 (+0.043) | 0.886 (+0.033) | 0.729 (+0.007) | 0.861 (+0.049) | 0.016 (+0.007) | 0.013 (+0.005) | 0.903 (+0.094) |  |
| Muon, 2× Muon lr | 4 | 11/13 | 0.725 (+0.064) | 0.727 (+0.016) | 0.865 (+0.011) | 0.717 (-0.005) | 0.841 (+0.029) | 0.013 (+0.004) | 0.011 (+0.004) | 0.871 (+0.062) | negative |
| **On Muon** | | | | | | | | | | | |
| balanced sampling (8 epochs, slowed) | 4 | 8 | 0.745 (+0.084) | 0.733 (+0.022) | 0.884 (+0.030) | 0.726 (+0.004) | 0.845 (+0.033) | 0.017 (+0.007) | 0.012 (+0.004) | 0.848 (+0.039) | confounded |
| balanced sampling, clean rerun | 4 | 13 | 0.768 (+0.107) | 0.734 (+0.023) | 0.884 (+0.030) | 0.720 (-0.002) | 0.850 (+0.037) | 0.018 (+0.009) | 0.013 (+0.006) | 0.899 (+0.090) | null |
| EMA 0.999 (seeds 7338, 1, 3) | 3 | 12/13 | 0.772 (+0.111) | 0.733 (+0.022) | 0.888 (+0.034) | 0.736 (+0.014) | 0.858 (+0.045) | 0.018 (+0.009) | 0.012 (+0.004) | 0.895 (+0.095) | null |
| EMA 0.9995 (seeds 7338, 1, 3) | 3 | 12/13 | 0.771 (+0.110) | 0.734 (+0.023) | 0.889 (+0.035) | 0.738 (+0.016) | 0.858 (+0.045) | 0.017 (+0.008) | 0.012 (+0.004) | 0.890 (+0.091) | null |
| **normalised targets (A3)** | 4 | 13 | 0.791 (+0.130) | 0.742 (+0.031) | 0.900 (+0.046) | 0.715 (-0.006) | 0.869 (+0.056) | 0.019 (+0.010) | 0.016 (+0.008) | 0.909 (+0.100) | **adopted** |
| **normalised targets + concat 9+12** | 4 | 13 | 0.798 (+0.137) | 0.765 (+0.054) | 0.901 (+0.047) | 0.713 (-0.009) | 0.873 (+0.061) | 0.019 (+0.010) | 0.015 (+0.007) | 0.911 (+0.102) | **best (4 seeds)** |
| **fast Muon** + normalised targets (seeds 7338, 1, 2) | 3 | 15 | 0.793 (+0.140) | 0.742 (+0.032) | 0.893 (+0.039) | 0.703 (-0.013) | 0.875 (+0.064) | 0.021 (+0.011) | 0.016 (+0.009) | 0.935 (+0.126) | same scores, 16% faster: **adopted** |
| fast Muon + normalised targets + concat 9+12 | 3 | 15 | 0.801 (+0.148) | 0.765 (+0.055) | 0.898 (+0.044) | 0.707 (-0.010) | 0.879 (+0.067) | 0.021 (+0.011) | 0.015 (+0.008) | 0.912 (+0.104) | **best config** |
| + brightness head (seeds 7338, 1, 2) | 3 | 14 | 0.790 (+0.136) | 0.743 (+0.034) | 0.890 (+0.036) | 0.725 (+0.008) | 0.859 (+0.048) | 0.019 (+0.009) | 0.014 (+0.007) | 0.891 (+0.083) | mixed: lesion up, age down |
| + brightness head, concat 9+12 | 3 | 14 | 0.796 (+0.143) | 0.765 (+0.055) | 0.894 (+0.040) | 0.728 (+0.011) | 0.861 (+0.050) | 0.019 (+0.009) | 0.014 (+0.007) | 0.898 (+0.090) | mixed |
| 40-epoch AdamW reference (~2.5 h, 1 seed, last layer) | 1 | 40 | 0.722 | 0.735 | 0.883 | 0.743 | 0.840 | 0.014 | 0.012 | 0.875 | upstream |

## Code

- **`optimizer: muon`, `muon_lr_mult`, `muon_impl: torch|batched`** (`train.py`, `utils/misc.py`: `MuonAdamW`, `BatchedMuon`). Muon covers the matrices named in `misc.MUON_PREFIXES`; AdamW covers the rest.
- **`norm_target`** (`model.py`). Per-patch normalized reconstruction targets. It is a `forward()` flag, so the architecture and eval are unchanged.
- **`model_kwargs.stats_head`, `stats_weight`** (`model.py`). Two extra decoder outputs per patch for its intensity mean and log std. Needs `norm_target`.
- **`mask_groups`** (`model.py`). Structured masking groups. `null` keeps the original sampler, bit-identical.
- **`modality_power`** (`train.py`: `ModalitySampler`). Scan-type balanced sampling with a fixed per-epoch quota. `null` keeps `shuffle=True`.
- **`ema_decays`** (`utils/misc.py`: `ModelEMA`). EMA copies saved as `checkpoint-ema<decay>.pth`.
  - Note: the timm-style decay warmup `min(d, (1+t)/(10+t))` binds until t≈9/(1−d), so 0.9998 never takes effect in a ~19k-step run.
- **`lr_decay`, `time_budget`** (`train.py`, `utils/misc.py`). Linear decay, and a schedule keyed to wall time.
- **`src/nanobrain/readout.py`.** `ReadoutViTMAE3D` reads out a concat or mean of blocks for the unmodified eval: `python -m nanobrain.readout CKPT OUT --layers 9 12`.
- **`scripts/arm_1hr.sbatch`.** One seed of one arm. The partition and account in its header are specific to our cluster.

Every option defaults to the original behaviour.

## Files here

- `results.csv`: one row per run, seed, readout and task, with the headline metric and its 95% CI.
  - Scalar metrics only. The eval JSONs' per-subject targets and predictions are not published.
- `runs.csv`: one row per run and seed, with the config overrides, epochs completed, final train loss and throughput.
- `collect_results.py`: rebuilds both CSVs from a run output directory.

## Open questions

- **Is a readout defined in the model class a fair submission?** That is `dense_embed`/`global_embed` over layers 9 and 12. The eval code itself is untouched.
- **Lesion:** what recovers it without costing the structure tasks?
