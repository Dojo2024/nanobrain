"""Paper-style figure: frozen linear-probe score at every encoder layer of the plain model
(Muon + normalized targets, ViT width 384). Line: seed 7338, all 12 layers. Gray: mean ± sd over
4 seeds at layers 3, 6, 9, 12. Writes <out>.pdf (vector) and <out>.png.
usage: plot_layerwise_paper.py <out stem>"""

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, "/admin/home/dojo/nanobrain/src")
from nanobrain.eval.tasks import HEADLINE_METRICS as H  # noqa: E402

O = Path("/data/dojo/nanobrain/output")
RUN = "a17/batched-nt"
SEED = 7338
SEEDS = [7338, 1, 2, 3]
OUT = Path(sys.argv[1])
TASKS = [("fomo_task3_t1w_hippocampus", "Hippocampus", "Dice"), ("fomo_task5_t1w_cortex", "Cortex", "Dice"),
         ("fomo_task2_dwi_tumor", "Tumor", "AUROC"), ("fomo_task1_adc_lesion", "Lesion", "AUROC"),
         ("fomo_task3_t1w_age", "Age", r"$R^2$"), ("fomo_task5_t1w_pmg", "PMG", "AUROC"),
         ("fomo_task4_t2w_nerve", "Nerve", "AP"), ("fomo_task4_t2w_vessel", "Vessel", "AP")]
BLUE, INK, INK2, MUTED, GRID, AXIS = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e6e5df", "#9b9a93"


def score(seed, layer, task):
    sub = "eval" if layer == 12 else f"readout/l{layer}"
    return json.load(open(O / RUN / f"s{seed}" / sub / f"{task}.json"))[H[task]]


plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "pdf.fonttype": 42,
})
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.3), sharex=True)
for i, (ax, (task, name, metric)) in enumerate(zip(axes.flat, TASKS)):
    layers = list(range(1, 13))
    ys = [score(SEED, layer, task) for layer in layers]
    peak = max(range(12), key=lambda k: ys[k])
    ax.axvline(layers[peak], color=GRID, lw=3, zorder=0)
    ax.plot(layers, ys, color=BLUE, lw=1.3, zorder=3)
    ax.plot(layers, ys, "o", ms=2.6, color=BLUE, zorder=4)
    ax.plot(layers[peak], ys[peak], "o", ms=5.5, mfc="white", mec=BLUE, mew=1.2, zorder=5)
    # 4-seed mean ± sd where all seeds were probed
    for layer in (3, 6, 9, 12):
        vals = [score(s, layer, task) for s in SEEDS]
        ax.errorbar(layer + 0.28, st.mean(vals), yerr=st.stdev(vals), fmt="s", ms=2.6, color=INK2,
                    elinewidth=0.8, capsize=1.5, capthick=0.8, zorder=2)
    ax.set_title(f"({chr(97 + i)}) {name}", loc="left", color=INK, pad=3)
    ax.text(1.0, 1.015, metric, transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=7.5)
    ax.set_xticks([1, 3, 6, 9, 12])
    ax.set_xlim(0.4, 12.8)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK2, labelsize=7)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
for ax in axes[1]:
    ax.set_xlabel("Encoder layer", color=INK2)

handles = [Line2D([0], [0], color=BLUE, lw=1.3, marker="o", ms=2.6, label="Seed 7338, every layer"),
           Line2D([0], [0], color=INK2, lw=0, marker="s", ms=2.6, label="4 seeds, mean ± sd"),
           Line2D([0], [0], color=BLUE, lw=0, marker="o", ms=5.5, mfc="white", mew=1.2, label="Best layer")]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01),
           fontsize=7.5, labelcolor=INK, handletextpad=0.4, columnspacing=1.6)
fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.0, w_pad=0.8)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=300)
print("saved", OUT.with_suffix(".pdf"), OUT.with_suffix(".png"))
