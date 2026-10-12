"""Frozen linear-probe score at every encoder layer (1-12), plain vs HC vs mHC, seed 7338.
Writes <out>.png and <out>.csv (the plotted numbers, as a table view).
usage: plot_every_layer.py <out stem>"""

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/admin/home/dojo/nanobrain/src")
from nanobrain.eval.tasks import HEADLINE_METRICS as H  # noqa: E402

O = Path("/data/dojo/nanobrain/output")
OUT = Path(sys.argv[1])
SEED = 7338
LAYERS = list(range(1, 13))
# categorical slots 1-3 (validated: CVD dE 9.2, normal 27.6); aqua is < 3:1 on the surface, so every
# series also has its own marker shape and the numbers go to a CSV table
MODELS = [("a17/batched-nt", "Plain", "#2a78d6", "o"),
          ("l1/dhc4-w384-matched", "Hyper-connections ×4", "#eb6834", "s"),
          ("l1/mhc4-w384", "mHC ×4", "#1baf7a", "^")]
TASKS = [("fomo_task3_t1w_hippocampus", "Hippocampus", "Dice"), ("fomo_task5_t1w_cortex", "Cortex", "Dice"),
         ("fomo_task2_dwi_tumor", "Tumor", "voxel AUROC"), ("fomo_task1_adc_lesion", "Lesion", "voxel AUROC"),
         ("fomo_task3_t1w_age", "Brain age", "R²"), ("fomo_task5_t1w_pmg", "PMG", "AUROC"),
         ("fomo_task4_t2w_nerve", "Nerve", "AP"), ("fomo_task4_t2w_vessel", "Vessel", "AP")]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def score(run, layer, task):
    sub = "eval" if layer == 12 else f"readout/l{layer}"
    p = O / run / f"s{SEED}" / sub / f"{task}.json"
    return json.load(open(p))[H[task]] if p.exists() else math.nan


rows = [(name, label, layer, score(run, layer, task)) for task, name, _ in TASKS for run, label, _, _ in MODELS
        for layer in LAYERS]
with open(OUT.with_suffix(".csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "model", "layer", "score"])
    w.writerows(rows)
missing = [(n, l, layer) for n, l, layer, s in rows if math.isnan(s)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10})
fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), facecolor=SURFACE)
for ax, (task, name, metric) in zip(axes.flat, TASKS):
    ax.set_facecolor(SURFACE)
    for run, label, color, marker in MODELS:
        ys = [score(run, layer, task) for layer in LAYERS]
        ax.plot(LAYERS, ys, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax.plot(LAYERS, ys, marker, ms=6, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.set_title(name, loc="left", color=INK, fontweight="bold")
    ax.text(1.0, 1.02, metric, transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=8.5)
    ax.set_xticks([1, 3, 6, 9, 12])
    ax.set_xlim(0.5, 12.5)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3)
for ax in axes[1]:
    ax.set_xlabel("encoder layer", color=INK2)

handles = [Line2D([0], [0], color=c, lw=2, marker=m, ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=l)
           for _, l, c, m in MODELS]
fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 0.975), labelcolor=INK)
fig.suptitle("Frozen linear-probe score at every encoder layer", x=0.01, y=0.975, ha="left", fontsize=12.5,
             color=INK, fontweight="bold")
fig.text(0.01, 0.915, "3D MRI masked autoencoder, ViT width 384, Muon, equal training epochs. One seed (7338); "
         "layers 3, 6, 9 and 12 also have 4-seed results.", ha="left", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUT.with_suffix(".png"), dpi=160, facecolor=SURFACE)
print("saved", OUT.with_suffix(".png"), OUT.with_suffix(".csv"))
print("missing cells:", len(missing), sorted({(l, layer) for _, l, layer in missing}))
