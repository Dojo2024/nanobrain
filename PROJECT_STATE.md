# nanobrain 1 h ablations: project state

Last updated 2026-10-13 ~07:15. Read this whole file before running anything. Every number for every
run is in [experiments/dojo_1h/all_results.md](experiments/dojo_1h/all_results.md). Regenerate it with
`summarize.py` (section 4) after new results land.

## 1. Goal and how ideas are judged

- **What we train.** Pretraining for clane9's nanobrain, a 3D brain-MRI ViT masked autoencoder (MAE),
  on FOMO300K.
  - Input: one 192×240×192 grid at 1 mm.
  - Default model: ViT-S (12 blocks, width 384) with 8 mm patches, a 24×30×24 token grid.
- **The budget is 1 h of wall clock on one H100.**
  - Training runs a 40-epoch cosine schedule, cut by `timeout 1h`, so the learning rate never decays.
  - An epoch is 46,592 images (1,456 steps × 32).
  - Judge every idea at 1 h. Runs at 40 epochs are diagnostics only.
- **We are still learning, so test every idea.** Never drop an idea because it "won't work on our data".
  Experiments decide.
- **There are two evals.** Compare each against a different baseline:

  | Eval | What it is | Compare against |
  |---|---|---|
  | **FOMO** (`src/nanobrain/eval/`, frozen, **never edit it**) | 8 tasks, one headline metric each: lesion (voxel AUROC), tumor (voxel AUROC), age (R²), hippocampus (dice), nerve (AP), vessel (AP), PMG (AUROC), cortex (dice). Two scores per run: the last layer (`eval/`) and a concatenation of layers 9 and 12 (`readout/cat9-12/`; `cat6-8` at depth 8). | **Our own control, paired by seed.** Report the mean difference and how many seeds improved, e.g. "+.010 (3/4)". Adopt only with 4 seeds. |
  | **brainmarks-smri** (MedARC, 12 tasks) | ADHD fails upstream, so 11 tasks are scored. Whole-scan tasks: ABIDE, CNP, UCSF IDH, UPENN 1-year survival, SOOP mRS (AUROC); IXI and OpenBHB age (R²). Segmentation tasks: BraTS whole tumor (WT), tumor core (TC) and enhancing tumor (ET), and SOOP lesion (dice). 300 subjects per split. | **The 5 reference models** (Neuro-JEPA, NeuroVFM, BrainDINO, DINOv3, BrainIAC), by mean rank when our model is added alone (1 = best of 6). Not against our 8 mm control. Per-task paired differences are still useful (see `paired.py`). |

- **Noise and caveats.**
  - FOMO PMG is noisy: 48 scans and a field-of-view confound.
  - FOMO nerve and vessel sit near the floor (AP ~.002–.024).
  - On brainmarks, whole-scan classification is noisy. Mean ranks move by ~0.1–0.3 on noise alone, so read the per-task paired differences.
  - CNP and SOOP overlap with the pretraining data.

## 2. Where things live

| What | Where |
|---|---|
| Repo | `/admin/home/dojo/nanobrain`, branch `1h-ablations` |
| Remotes | `dojo` = github.com/Dojo2024/nanobrain (the user's fork; push here); `origin` = MedARC-AI/nanobrain (**never push or open PRs**) |
| Model code | `src/nanobrain/model.py` (`ViTMAE3D`, patch size, `to_eval_grid`, P14 `decoder_sub`, `layout_head`, L8 `global_loss` + `sigreg`) |
| Training | `src/nanobrain/train.py`, config `src/nanobrain/config/default_pretrain.yaml` (every knob, with defaults) |
| Layer readout | `src/nanobrain/readout.py` |
| 1 h launcher | `scripts/arm_1hr.sbatch`: train 1 h, then FOMO one task per call, then the cat9-12 readout, then brainmarks if `BRAINMARKS=1` |
| Brainmarks launcher | `scripts/brainmarks.sbatch`; adapter `experiments/brainmarks/nanobrain_adapter.py` |
| Brainmarks clone | `~/brainmarks-smri`, upstream 20c2964, unmodified; data in `/data/connor/brain_datasets/datasets` |
| Run outputs | `/data/dojo/nanobrain/output/<idea>/<arm>/s<seed>/` (`config.yaml`, `log.json`, `checkpoint-last.pth`, `eval/`, `readout/<name>/`, `brainmarks*/`, `*-failed.txt`) |
| Result tables | `experiments/dojo_1h/all_results.md` (generated), `summarize.py`, `paired.py` |
| Checks | `experiments/dojo_1h/patchsize/check_*.py` (CPU numerical checks of the eval-grid interpolation, the sub-token decoder and SIGReg) |
| Full history | `/admin/home/dojo/nanobrain-ideas.md` (1,668-line ledger, outside the repo). Brainmarks and patch size are at ~lines 859–1040; the P1–P50 index is §3c. |
| Literature reports | `/admin/home/dojo/reports/Nanobrain new pretraining ideas.md` (L1–L20), `.../Nanobrain patch size and 50 ideas.md` (P1–P50) |
| Stale | `experiments/dojo_1h/README.md` is the 2026-10-05 write-up; its "best config" is superseded by section 5 |

Git state:
- Commits `bff48a5` (code) and `135bc1c` (experiments) are pushed to `dojo/1h-ablations`.
- The adapter's `brain_mode` and this file's scripts may still be uncommitted; check `git status`.

## 3. Rules (hard constraints from the user)

**Submitting jobs**
1. **Ask the user before every new `sbatch` or `salloc`.** Approval covers one submission only.
   - `srun --jobid=<ours> --overlap` for checks is fine.
   - **Install nothing on any node.**
2. **Don't crash the GPU nodes.**
   - Before submitting, and while jobs run, check the node's real free memory:
     `srun --jobid=<job> --overlap -n1 awk '/MemAvailable/{print int($2/1048576)" GB"}' /proc/meminfo`.
     `sinfo %e` leaves out page cache and is misleading.
   - Brainmarks segmentation jobs reach ~62 GB RSS. Use `--mem=96G`, or `128G` for an 8 mm w768 job that runs several brainmarks passes.
3. **Job names and limits.**
   - Name jobs `nb-<idea>-<arm>-s<seed>`.
   - Run at most 8 jobs at once.
   - Use account `training`, QOS `normal`.
   - **Never add arms the user did not ask for.**
4. **Don't touch other jobs,** e.g. `autoresearch-serve`. Run tests as their own jobs, never inside the user's holder jobs.

**Files and data**
5. **Ask before writing anywhere under `/data/dojo`,** except `/data/dojo/nanobrain/output` (approved).
6. **Never edit `src/nanobrain/eval/`.**
7. **Never reuse a run dir.** `arm_1hr.sbatch` refuses one that already has `checkpoint-last.pth`, since resuming breaks the 1 h budget. Use a new arm name. Don't delete run dirs without asking.

**Working on the login node**
8. The login node has no GPU, so don't benchmark there.
   - Cap CPU threads with `OMP_NUM_THREADS=2`.
   - Always use the repo's interpreter with `VIRTUAL_ENV=` unset: `VIRTUAL_ENV= .venv/bin/python ...`.

## 4. How to run and score

```bash
cd /admin/home/dojo/nanobrain

# One seed of a 1 h arm (train, FOMO, cat9-12 readout, brainmarks). Seeds: 7338 1 2 3.
BRAINMARKS=1 sbatch -J nb-<idea>-<arm>-s<seed> --mem=96G --time=2:30:00 \
    scripts/arm_1hr.sbatch <idea>/<arm> <seed> <config overrides...>
#   at depth 8 add READOUT_LAYERS="6 8"; avoid nodes that are low on memory with --exclude=

# brainmarks on an existing checkpoint (TASKS="a b" picks a subset)
sbatch -J nb-bm-<arm>-s<seed> --mem=96G --time=1:30:00 scripts/brainmarks.sbatch \
    <run dir>/checkpoint-last.pth <run dir> brainmarks [model_kwargs.brain_mode=pool|strip]

# every run as tables (mean over seeds, brainmarks ranks vs the 5 references)
OMP_NUM_THREADS=2 VIRTUAL_ENV= .venv/bin/python experiments/dojo_1h/summarize.py \
    /data/dojo/nanobrain/output > experiments/dojo_1h/all_results.md

# paired comparison, FOMO + brainmarks with a per-seed bootstrap
OMP_NUM_THREADS=2 PYTHONPATH=~/brainmarks-smri/src VIRTUAL_ENV= .venv/bin/python \
    experiments/dojo_1h/paired.py <arm> <control> [--seeds 7338 1] [--variant brainmarks-brainpool]
```

How long each piece takes:
- Training: 1 h.
- FOMO: ~6 min.
- The cat9-12 readout: ~12 min.
- Brainmarks: ~30 min.
- So one job takes ~1 h 50 min, plus queue time.

## 5. Current best recipes (all use the "Faster Muon" (FM) recipe)

FM = `base_lr=0.003 optimizer=muon muon_impl=batched norm_target=true`.

| Use | Arm | Overrides on top of FM | Epochs in 1 h |
|---|---|---|---|
| **Best on FOMO** (8 mm) | `n23/w768-d8` | `model_kwargs.embed_dim=768 model_kwargs.num_heads=12 model_kwargs.depth=8` (readout cat6-8) | 12 |
| Best 8 mm on brainmarks (1 seed) | `n20/vis1024-w768` | `model_kwargs.embed_dim=768 model_kwargs.num_heads=12 num_visible=1024` | 11 |
| **Best on brainmarks** (16 mm) | `ps16/w1024-sub2` | same as the next row but `model_kwargs.embed_dim=1024 model_kwargs.num_heads=16` | 18–21 |
| Previous 16 mm best | `ps16/w768-sub2` | `model_kwargs.embed_dim=768 model_kwargs.num_heads=12 patch_size=16 num_visible=168 num_predict=168 model_kwargs.decoder_sub=2 model_kwargs.decoder_sub_depth=1` | 20 |
| Simple 8 mm control | `n11/width512` | `model_kwargs.embed_dim=512 model_kwargs.num_heads=8` | 13 |

**8 mm runs are GPU-bound.** Width 512 gets ~13 epochs and width 768 ~11–12.

**16 mm runs are bound by the data loader.**
- They run at ~265–335 volumes/s with the GPU only 44–60% busy, which gives ~19–20 epochs.
- Extra width or decoder compute is therefore free in wall-clock time.
- Several 16 mm jobs on one node slow each other. Check the epochs column before comparing.
- More loader workers do not help (benchmark 2026-10-13: 15 workers = 8 workers). The wait is in storage reads or the single main process (pin memory, per-scan GPU preprocessing in a Python loop at `train.py:282`). Even with zero data wait the step would cap at ~450 vol/s.

## 6. Results

### 6a. FOMO, last layer (mean over seeds)

| Arm | Seeds | Epochs | Lesion | Tumor | Age | Hippocampus | Nerve | Vessel | PMG | Cortex |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` (Connor's recipe: AdamW 1e-3, w384) | 4 | 15 | .721 | .854 | .812 | .661 | .009 | .008 | .809 | .711 |
| `a17/batched-nt` (FM, w384) | 4 | 14 | .714 | .897 | .873 | .791 | .020 | .016 | .932 | .741 |
| `n11/width512` (FM w512) | 4 | 13 | .713 | .903 | .877 | .802 | .021 | .016 | .935 | .750 |
| `n20/vis1024-w768` | 4 | 11 | .712 | .910 | .882 | .809 | .022 | .016 | .933 | .754 |
| **`n23/w768-d8`** | 4 | 12 | .719 | .899 | .874 | .811 | .024 | .016 | .928 | .769 |
| `l1/adamw-nt-w384` (best lesion) | 4 | 14 | **.757** | .884 | .842 | .749 | .019 | .012 | .858 | .743 |
| `e40/width768` (40 epochs, diagnostic) | 2 | 40 | .707 | .908 | .897 | .825 | .023 | .017 | .921 | .766 |
| `ps16/baseline` (16 mm, Connor's recipe) | 4 | 20 | .668 | .865 | .795 | .549 | .002 | .003 | .819 | .627 |
| `ps16/muon` (16 mm, Muon lr 3e-3) | 4 | 19 | .706 | .910 | .833 | .618 | .003 | .004 | .859 | .649 |
| `ps/p16-w768` (16 mm, FM w768) | 4 | 20 | .681 | .923 | .860 | fails | .003 | .004 | .900 | .656 |
| `ps16/w768-sub2` (16 mm, + P14) | 4 | 20 | .691 | .919 | .860 | fails | .003 | .005 | .912 | .657 |
| **`ps16/w1024-sub2`** (16 mm, width 1024) | 4 | 20 | .702 | .926 | .866 | fails | .004 | .004 | .922 | .658 |

Best second readouts:
- `n23/w768-d8` cat6-8: cortex .790, hippocampus .815, lesion .719.
- `n20/vis1024-w768` cat9-12: tumor .913, age .885.

"fails" means the FOMO hippocampus l-bfgs probe does not converge at 16 mm w768. This is an eval-side failure (`eval-failed.txt`), not a training failure.

### 6b. brainmarks-smri vs the 5 reference models (mean over seeds)

| Model | Seeds | Rank | ABIDE | CNP | IDH | UPENN | mRS | IXI | BHB | WT | TC | ET | Lesion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Neuro-JEPA* | ref | | .63 | .68 | .94 | .74 | .72 | .87 | .88 | .75 | .59 | .43 | .35 |
| *NeuroVFM* | ref | | .59 | .70 | .94 | .72 | .73 | .92 | .84 | .69 | .58 | .44 | .38 |
| *BrainDINO* | ref | | .61 | .71 | .93 | .71 | .71 | .80 | .79 | .75 | .64 | .45 | .32 |
| *DINOv3* | ref | | .61 | .62 | .92 | .74 | .77 | .79 | .74 | .73 | .59 | .43 | .30 |
| *BrainIAC* | ref | | .52 | .55 | .76 | .65 | .71 | .58 | .47 | .74 | .31 | .32 | .37 |
| `baseline` 8 mm | 1 | 3.55 | .583 | .682 | .931 | .734 | .689 | .765 | .675 | .808 | .547 | .558 | .513 |
| `connor40` (Connor's 40-epoch run) | 1 | 2.73 | .626 | .669 | .927 | .729 | .708 | .810 | .750 | .822 | .592 | .610 | .521 |
| `a17/batched-nt` 8 mm FM | 1 | 2.77 | .612 | .646 | .902 | .732 | .698 | .871 | .804 | .834 | .662 | .664 | .527 |
| `n20/vis1024-w768` 8 mm | 1 | 2.68 | .614 | .669 | .917 | .736 | .701 | .864 | .820 | .852 | .704 | .688 | .556 |
| `n23/w768-d8` 8 mm | 1 | 2.73 | .608 | .663 | .919 | .737 | .694 | .857 | .795 | .850 | .711 | .688 | .543 |
| `ps16/baseline` 16 mm | 4 | 3.64 | .608 | .672 | .915 | .715 | .689 | .746 | .656 | .797 | .490 | .498 | .444 |
| `ps16/muon` 16 mm | 4 | 3.00 | .619 | .672 | .926 | .732 | .690 | .830 | .763 | .781 | .583 | .501 | .429 |
| `ps/p16-w768` 16 mm | 4 | 2.45 | .625 | .685 | .928 | .733 | .688 | .876 | .819 | .795 | .613 | .527 | .429 |
| `ps16/w768-sub2` 16 mm | 4 | 2.32 | .632 | .698 | .924 | .748 | .693 | .877 | .812 | .801 | .624 | .538 | .439 |
| **`ps16/w1024-sub2`** 16 mm | 4 | **2.14** | .636 | .705 | .926 | .738 | .693 | .886 | .823 | .807 | .638 | .548 | .451 |
| `ps16/sub2-infonce` 16 mm | 2 | 2.27 | .637 | .706 | .920 | .744 | .698 | .879 | .832 | .795 | .623 | .528 | .431 |

With our model added, Neuro-JEPA ranks ~2.45–2.73. Ranks are recomputed by `summarize.py`. Some differ by ±0.05 from older ledger values because of ties after rounding to 2 dp.

### 6c. What we have learned

1. **The FM recipe is the big win, and it transfers.**
   - On FOMO at 8 mm, it lifts every task except lesion: hippocampus +.13, PMG +.12, age +.06, tumor +.04.
   - It transfers to 16 mm: lesion +.038, tumor +.044, age +.038, hippocampus +.068 (all 3/4 or 4/4).
   - It transfers to brainmarks segmentation and age: TC .55→.66–.71, IXI/OpenBHB +.08–.13.
   - It does nothing for whole-scan classification.
2. **At 8 mm, encoder capacity is the limit, not the number of steps.** Width 512, 768 and 768 with depth 8 each win despite fewer epochs.
3. **Lesion is the open FOMO problem.**
   - Muon costs lesion. AdamW + normalized targets gives the best lesion (.757; HC-AdamW cat9-12 .774).
   - Every Muon-side fix failed: half the Muon lr, registers, intensity augmentation.
   - Under Muon, cortex and lesion peak by epoch ~10 and then decline at 40 epochs.
4. **Patch 12/16 mm loses on FOMO but wins on brainmarks.**
   - FOMO: thin structures suffer (hippocampus −.12/−.17, cortex −.07/−.10, nerve/vessel collapse), even with ~1.5× the epochs; only tumor gains.
   - Brainmarks: 16 mm w768 (+ the P14 sub-token decoder) ranks **1st of 6**. It lifts whole-scan classification toward the references and still leads them on WT/ET/lesion.
   - So the brainmarks segmentation lead is part 8 mm grid and part model: even the plain 16 mm baseline beats every reference on WT/ET/lesion.
5. **Whole-scan classification on brainmarks is flat across every 8 mm change.**
   - Only the 16 mm grid plus width moves it.
   - The global losses (L8) don't move it by the +.03–.05 needed.
   - InfoNCE trades segmentation for age and CNP.
6. **The hyper-connection family (HC/FC/mHC) gives small gains per epoch but loses per hour** to plain width at our scale.

## 7. Every idea tested so far

| Idea | Arm dir | Verdict |
|---|---|---|
| N7: fit the lr decay into 1 h | `n7/*` | negative |
| N8: higher lr (3e-3, default schedule) | `n8/lr3e-3-cut40` | **adopted** |
| N1: concatenated readout of layers 9+12 | `readout/cat9-12` | **adopted** as the second score |
| N3: class token in the global embedding | (readout) | null |
| B1 block / B2 axial-tube masking | `b1/*`, `b2/*` | negative (tubes: lesion +.04, everything else lost); masking retired |
| A17: Muon | `a17/muon` | **adopted** |
| Muon lr 2× / 0.5× | `a17/muon-lr2x`, `n8/muonlr05-w512` | negative |
| B26: modality-balanced sampling | `b26/sqrt-r2` | null |
| N16: EMA | `n16/ema` | null |
| A3: normalized targets | `a3/normtarget` | **adopted** |
| Batched Muon | `a17/batched-nt` | **adopted** (faster, same scores) |
| N17: stats head | `n17/statshead` | mixed (lesion up, age down) |
| N18: narrow decoder | `n18/dec10x192` | negative |
| N11: width 512 / 640 / 768 | `n11/*` | 512 **adopted**; 640/768 trade-offs |
| N19: cross-attention decoder | `n19/xdec` | mixed |
| L1: HC / FC / mHC at w384 | `l1/*`, `e40/*c4-w384` | null per hour |
| AdamW + normalized targets | `l1/adamw-nt-w384` | best lesion, loses elsewhere |
| N9: one mask per image | `n9/onemask-w512` | cancelled (loader-throughput confound) |
| N20: 1024 visible patches | `n20/*` | w768 positive; w512 null |
| N21: 2-epoch warmup | `n21/wu2-vis1024-w768` | negative |
| N22: intensity augmentation / 4 registers | `n22/*` | negative / retired |
| N23: width 768, depth 8 | `n23/w768-d8` | **best 8 mm on FOMO** |
| E40: 40-epoch wave | `e40/*` | diagnostic (width 768 wins at 40 epochs) |
| Plain patch 12/16 | `ps/p12`, `ps/p16`, `ps/p16-ns8` | negative on FOMO; brainmarks 2.68/2.77/3.05 |
| 16 mm + width 768 | `ps/p16-w768` | brainmarks 2.45 |
| 16 mm ladder | `ps16/baseline`, `ps16/muon` | the recipe transfers |
| P14: sub-token decoder (16 mm token decoded as 8× 8 mm) | `ps16/w768-sub2` | small consistent positive |
| P14: layout head (octant mean / log std) | `ps16/w768-layout` | mixed (FOMO lesion +.038, PMG −.011) |
| L8: SIGReg β 0.1 / 1, InfoNCE | `ps16/sub2-sigreg-b0.1`, `-b1`, `-infonce` | not adopted |
| Loader workers 8 / 12 / 15 at 16 mm (benchmark, job 530044) | `experiments/dojo_1h/loader/` (outputs deleted) | **negative**: 252 / 196 / 255 vol/s, data wait .035 / .048 / .030 s per ~.105 s step; worker count is not the limit (I/O or the main process is); 4-seed arm not run |
| P20: brain-only readout, `pool` | `brainmarks-brainpool/` | small, not significant: CNP/OpenBHB up at both grids; rank 2.09 vs 2.32 (16 mm), 2.73 vs 2.86 (8 mm), same 2 seeds |
| P20: brain-only readout, `strip` | `brainmarks-brainstrip/` | negative: age −.04 (16 mm, 2/2 significant), classification slightly down, lesion +.012; rank 2.73 vs 2.32 |
| Width 1024 at 16 mm | `ps16/w1024-sub2` | **positive, best 16 mm**: brainmarks rank 2.14 vs 2.32 (WT/TC 4/4), FOMO lesion +.011, tumor/age +.007 (4/4); free in wall-clock |
| Whole-scan readouts on width 1024: brain pool, layers 9+12 | `brainmarks-brainpool/`, `brainmarks-cat9-12/` | **null**, keep the plain readout: rank 2.18 / 2.23 vs 2.14; pool CNP +.011 (4/4, the one effect seen in all 3 P20 tests); layers 9+12 CNP/IDH/UPENN +.005–.006 (4/4) but ABIDE −.016 (0/4) |
| A2: spectral loss (SL-MAE, per patch) | `a2/spec0.1-w512`, `a2/spec0.5-w512` | **null to negative, not adopted**: FOMO tumor −.009/−.011, PMG −.020/−.013, hippocampus −.003/−.009 (0/4 at 0.5), thin structures flat; brainmarks rank 2.73 / 2.86 (control 2.73), 0.5 costs segmentation (WT −.005, 2/2 significant) |
| P33: InfoNCE switched on at 45 min, on width 1024 | `p33/w1024-infonce-late45` | **negative**: brainmarks rank 2.36 vs 2.14; CNP −.014 (0/4), IXI −.014 (0/4), OpenBHB −.015 (1/4), IDH/UPENN/mRS and segmentation ±.005; FOMO PMG −.036 (0/4), lesion −.007 (cat9-12 −.019). Switching InfoNCE on late hurts the whole-scan tasks it targets (on from the start at w768 it gave OpenBHB +.028) |
| SiamJEPA / SiamJEPA + RST, default width, 16 mm | `siam/w384`, `siam/w384-rst` | **strongly negative, retired as a pure objective**: brainmarks rank 5.36 / 5.45 (MAE control `ps16/muon` 3.00); vs the control on FOMO hippocampus −.22 / −.32, age −.13 / −.29, PMG −.19 / −.44, cortex −.09 / −.13, tumor −.08 / −.13 (all 0/4); brainmarks IXI R² .54 / .35 (control .83), OpenBHB .52 / .39 (.76), TC .33 / .37 (.58), ET .27 / .34 (.50). RST is worse still. Final cosine loss .11 without RST (prediction became easy), ~.86 with RST |
| SiamJEPA at 40 epochs (diagnostic, user) | `siam/e40-w384`, `siam/e40-w384-rst` | **negative, no gain from training longer** (2 seeds; 530544–47). Plain is flat from epoch 10 to 40 (hippocampus .41/.43/.40/.38, age .70/.72/.69/.72, cortex .58/.57/.55/.56; tumor .81 → .84); RST gets worse with training (hippocampus .33 → .23, age .60 → .48, tumor .78 → .72). Brainmarks rank 5.27 (1 h: 5.18) and 5.64 (1 h: 5.50) on seeds 7338/1; MAE 1 h `ps16/muon` 3.00. SiamJEPA retired |
| P7: 16 mm for 30 min, then 8 mm (w512) | `p7/w512-switch30` | **negative**: 10–11 epochs at 16 mm + 5–6 at 8 mm; FOMO vs `n11/width512` all down (hippocampus −.014, cortex −.010, tumor −.015, PMG −.040; 0–1/4); brainmarks rank 2.77 (control 2.73), segmentation −.015 to −.033 (2/2 significant), whole-scan flat. The switch itself works (177–179 vol/s after it) |
| P14 sub4: 16 mm with 4x sub-token decoder (`decoder_sub=4`) | `ps16/w1024-sub4` | **negative**: 10,752 sub-tokens in decoder self-attention cut throughput to 138 vol/s (8–9 epochs vs 20–21 in 1 h); Brainmarks rank drops, BraTS TC -0.032, ET -0.022, IXI Age -0.034 (all -4/4 seeds worse) |
| Async CUDA stream prefetching (`prefetch_batch_generator`) | `ps16/w1024-sub2-async` | **neutral to positive**: 18–19 epochs, fully overlaps host-to-device transfers and GPU pre-processing; Brainmarks rank ~2.14 preserved (ABIDE +.001, IDH +.002, UPENN +.002, OpenBHB +.003, BraTS WT +.002) |

## 8. Latest round (submitted 2026-10-13, all COMPLETED by ~07:10; nothing running)

| Jobs | Name | What | Output |
|---|---|---|---|
| 529901–02 | `nb-p20-p16sub2-s{7338,1}` | P20 on `ps16/w768-sub2`: `pool` pass (7 whole-scan tasks), then `strip` (11 tasks) | `<run>/brainmarks-brainpool/`, `<run>/brainmarks-brainstrip/` |
| 529903–04 | `nb-p20-w768d8-s{7338,1}` | Same on `n23/w768-d8`. s1 also runs its missing plain `brainmarks/` control | same |
| 529905–08 | `nb-ps16-w1024-s{7338,1,2,3}` | `ps16/w1024-sub2`: the `ps16/w768-sub2` recipe at `embed_dim=1024 num_heads=16`, `BRAINMARKS=1` | `ps16/w1024-sub2/s*/` |

**P20 is a readout change only; the checkpoints are reused.** It adds `brain_mode` to the brainmarks adapter, using the `brain_mask` that brainmarks provides for every sample:
- `pool`: the input is unchanged, but the whole-scan embedding averages only patches that contain brain.
- `strip`: zero outside the brain and z-score inside it, so the model sees skull-stripped inputs, as the references do.
- `none`: today's behaviour.

On an IXI scan, cos(none, pool) = .926 and cos(none, strip) = .885. So the modes really change the embedding.

**P20 result (06:15, 2 seeds each, paired against the plain readout on the same seeds):**
- `pool`, 16 mm: CNP +.014, OpenBHB +.014, everything else within ±.004. No seed's paired CI excludes 0. Rank 2.09 vs 2.32.
- `pool`, 8 mm `n23/w768-d8`: CNP +.031, OpenBHB +.011, IXI −.015. Rank 2.73 vs 2.86.
- `strip`, 16 mm: IXI −.044 and OpenBHB −.041 (both seeds significant), CNP −.022, ABIDE −.013; lesion +.012 and WT +.006 (2/2). Rank 2.73.
- `strip`, 8 mm: IXI −.035, OpenBHB −.018, lesion +.011. Rank 3.00.
- Reading: our models were trained on whole heads, so skull-stripped inputs are out of distribution and age loses the head context. Brain-only pooling is cheap and leans positive but is inside the noise.

**Width 1024 (FOMO, 4 seeds vs `ps16/w768-sub2`):** 18/20/21/21 epochs at 272–299 vol/s (same as width 768: loader-bound, so the width is free). Last layer: lesion +.011 (3/4), tumor +.007 (4/4), age +.007 (4/4), PMG +.010 (2/4), cortex +.001. cat9-12: lesion +.024 (3/4). Hippocampus fails to converge, as at width 768. 
**Width 1024 brainmarks (4 seeds vs `ps16/w768-sub2`, seeds better):** rank **2.14** vs 2.32. WT +.006 (4/4), TC +.013 (4/4), ET +.010 (3/4), lesion +.012 (3/4), IXI +.008 (3/4), OpenBHB +.012 (2/4), CNP +.007 (3/4), UPENN −.010 (1/4), rest ±.003. Small but consistent, and free because the run is loader-bound.

**How to score when done.**
- `summarize.py` gives the brainmarks rank of the `brainpool` and `brainstrip` rows. Pool rows copy the segmentation cells from the same seeds' plain run.
- P20 pairs:
  - `paired.py ps16/w768-sub2 ps16/w768-sub2 --seeds 7338 1 --variant brainmarks-brainpool`
  - the same with `--variant brainmarks-brainstrip`
  - the same two for `n23/w768-d8`
- Width 1024: `paired.py ps16/w1024-sub2 ps16/w768-sub2` (FOMO and brainmarks), plus its rank vs the references.
- Then add the results to section 7, and to the ledger `/admin/home/dojo/nanobrain-ideas.md` (end of the "Patch size 12 / 16 mm" section).

**Readout companion on `ps16/w1024-sub2`, 4 seeds, no training (jobs 530064–67, COMPLETED 08:15): NULL.** Pool vs plain: CNP +.011 (4/4), OpenBHB +.005, IXI −.009 (0/4), rest ±.002; rank 2.18. Layers 9+12 vs plain: CNP +.006, IDH +.005, UPENN +.006 (all 4/4), IXI +.002, ABIDE −.016 (0/4), mRS −.005, OpenBHB −.004; rank 2.23 (plain 2.14). Readout changes move whole-scan tasks by ~±.01, far below the gap to the references (IDH .94, mRS .77, age .92/.88); that gap is in training, not the readout.

Setup:
Jobs 530064–67 `nb-bmro-w1024-s{7338,1,2,3}` run `experiments/dojo_1h/readout/bm_global_readouts.sbatch <run dir>` on the 7 whole-scan tasks:
- `brainmarks-brainpool/`: `brain_mode=pool` on checkpoint-last.
- `brainmarks-cat9-12/`: the layers 9+12 readout (rebuilt with `nanobrain.readout --layers 9 12`, then deleted); the global embedding is mean-pooled layer 9 and layer 12 tokens, concatenated.

Score with `paired.py ps16/w1024-sub2 ps16/w1024-sub2 --variant brainmarks-brainpool` (or `brainmarks-cat9-12`); `summarize.py` fills their segmentation cells from the plain run for the rank. Not run: pool combined with layers 9+12.

**A2 spectral loss smoke test (job 530070, 2026-10-13 07:53): runs.** One epoch of `n11/width512` + `spectral_weight=0.1` at 175 vol/s (control 172–180), bf16 + torch.compile fine (Inductor runs the complex FFT eagerly, no cost). `measure.py` on 8 real batches:
- smoke checkpoint (1 epoch, w=0.1): masked MSE .887, spectral .634
- `n11/width512` s7338 (1 h, trained without the term): MSE .412, spectral .940

A fully trained MSE-only model still has a large spectral error, so the term asks for something MSE does not give. At a trained MSE of ~.4, weight 0.1 makes it ~15–23% of the loss and 0.5 about equal to the MSE.

**Running (submitted 2026-10-13 ~08:30): A2 spectral loss, 8 jobs.** `BRAINMARKS=1 sbatch --mem=96G --time=2:30:00 scripts/arm_1hr.sbatch a2/spec<w>-w512 <seed> base_lr=0.003 optimizer=muon muon_impl=batched norm_target=true model_kwargs.embed_dim=512 model_kwargs.num_heads=8 spectral_weight=<w>`, w ∈ {0.1, 0.5}, seeds 7338 1 2 3 (jobs 530150–53 = 0.1, 530154–57 = 0.5). Score with `paired.py a2/spec<w>-w512 n11/width512`; the control has brainmarks on s7338 and s1 only, so brainmarks pairs on those 2 seeds (all 4 seeds still get a rank vs the references).

**A2 RESULT (2026-10-13 ~10:40), 4 seeds vs `n11/width512`: null to negative, not adopted.** 12–13 epochs (172 vs 177 vol/s). FOMO last layer, w=0.1 / 0.5: lesion −.003 / +.004, tumor −.009 (1/4) / −.011 (1/4), age −.005 (0/4) / −.004, hippocampus −.003 / −.009 (0/4), PMG −.020 (0/4) / −.013, cortex, nerve and vessel ±.001. cat9-12 at 0.5: lesion +.011 (3/4), hippocampus −.010 (0/4). Brainmarks (2 control seeds): rank 2.73 / 2.86 (control 2.73); IDH +.013 / +.009, mRS +.009 / +.010, OpenBHB +.019 / +.011, IXI −.019 / −.022; at 0.5 segmentation drops (WT −.005, 2/2 significant). Training lowers the spectral error (~.94 → ~.4) but the frozen probes do not gain; the paper's gains were fine-tuned, 300 epochs, full-volume.

**Running (submitted 2026-10-13 ~10:35): P33 and P7, 8 jobs, `BRAINMARKS=1`, 96G, 2:30.**
- P33 `p33/w1024-infonce-late45` (530271–74): the `ps16/w1024-sub2` recipe + `model_kwargs.global_loss=infonce global_weight=0.1 global_start_min=45`. Compare with `ps16/w1024-sub2` (and `ps16/sub2-infonce`, on from the start, at w768).
- P7 `p7/w512-switch30` (530275–78): FM w512 at `patch_size=16 num_visible=168 num_predict=168 model_kwargs.decoder_sub=2 model_kwargs.decoder_sub_depth=1 switch_min=30`; it continues at 8 mm (1280/1280, plain decoder) from the first epoch boundary after 30 min. `log.txt` says `switched to patch 8` at the switch. The final checkpoint is an 8 mm model. Compare on FOMO with `n11/width512` (and `n23/w768-d8`), on brainmarks with the 5 references (and `n11/width512`, `ps/p16`).
- Code: `train.switch_patch_size`, `train.make_optimizer`, `model.fine_patch_state_dict`; CPU checks `experiments/dojo_1h/patchsize/check_switch.py`.

## 9. Offered, not yet approved by the user

- P13: an octant readout head.
- Layout head, seeds 2 and 3.
- InfoNCE: 2 more seeds, or a late switch-on (P33).
- The remaining 16 mm ladder steps (normalized targets, width) as separate 4-seed steps.
- A slimmability census.
- A2: a spectral loss.
- A note to Connor about the ADHD upstream bug: `datasets.py:165` `desc_pattern r"(acq-\d_)?run-1"`, but `adhd200/images.tsv` has desc `n/a`.
- Untested literature ideas: P1–P50 (§3c of the ledger) and L1–L20.

## 10. Traps

- **Coarse patches and the frozen FOMO eval.**
  - The frozen eval asserts the 8 mm grid, so 12/16 mm models go through `to_eval_grid` (trilinear, mask-weighted) inside `dense_embed`.
  - brainmarks and `global_embed` use `dense_embed_native` instead, the model's own grid.
  - `readout.py` overrides `dense_embed_native`.
- **Losses that only exist in training.** `train.py`'s end-of-epoch plot calls the model in eval mode with `num_samples=1`. Gate any such loss on `self.training`; the first L8 submission crashed on exactly this.
- **Zero-initialized decoder head.** Its encoder gradient is zero at step 1. CPU checks need ≥2 optimizer steps.
- **brainmarks class keys.** Classification probability keys are JSON spellings (`"true"`) of bool classes.
- **Deleted checkpoints.** `b1/block4`, `b2/tube-si` and `b2/tube-si-lr5e-3` lost `checkpoint-last.pth` in the 2026-10-10 cleanup. Their eval JSONs remain.
- **One CPU core for overlap steps.** Several `srun --overlap` steps all pin to the job's first CPUs. Use one step plus `taskset`.
