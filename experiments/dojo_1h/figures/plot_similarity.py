"""Plot layer similarity (layer_similarity.py output): profiles like HC paper Fig. 3, and CKA heatmaps."""

import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

O = Path("/data/dojo/nanobrain/output")
OUT = Path(sys.argv[1])
SEEDS = [7338, 1, 2, 3]
ROWS = [("Muon", "a17/batched-nt", "l1/dhc4-w384-matched"), ("AdamW", "l1/adamw-nt-w384", "l1/dhc4-adamw-w384-matched")]
PLAIN, HC = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load(run):
    return [json.loads((O / run / f"s{s}" / "similarity.json").read_text()) for s in SEEDS]


def band(results, key):
    """mean over seeds of each position's [5th, 50th, 95th] percentile"""
    n = len(results[0][key])
    return [[st.mean(r[key][i][q] for r in results) for i in range(n)] for q in range(3)]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=3)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10})
PANELS = [
    ("cos_sub", "Input vs previous sublayer's input", "cosine", "sublayer (attention, MLP alternating)"),
    ("cos_block", "Block output vs previous block output", "cosine", "block (stream sum for HC)"),
    ("delta", "Change made by each block", "relative size", "block (stream sum for HC)"),
]
fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8), facecolor=SURFACE)
data = {}
for row, (opt, plain, hc) in enumerate(ROWS):
    data[opt] = {"Plain": load(plain), "Hyper-connections ×4": load(hc)}
    for col, (key, title, metric, xlabel) in enumerate(PANELS):
        ax = axes[row, col]
        style(ax)
        for label, color in [("Plain", PLAIN), ("Hyper-connections ×4", HC)]:
            lo, mid, hi = band(data[opt][label], key)
            x = list(range(2, len(mid) + 2)) if key == "cos_sub" else list(range(1, len(mid) + 1))
            ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0)
            ax.plot(x, mid, color=color, lw=2, solid_capstyle="round")
        if key == "delta":
            ax.set_yscale("log")
        ax.set_title(f"{opt}: {title}", loc="left", color=INK, fontweight="bold", fontsize=9.5)
        ax.text(1.0, 1.02, metric, transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=8.5)
        if row == 1:
            ax.set_xlabel(xlabel, color=INK2)
handles = [Line2D([0], [0], color=c, lw=2, label=l) for l, c in [("Plain", PLAIN), ("Hyper-connections ×4", HC)]]
handles.append(plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.25, lw=0, label="5th–95th percentile over tokens"))
fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.99, 0.975), labelcolor=INK)
fig.suptitle("How similar consecutive layers are: plain vs hyper-connections", x=0.01, y=0.975, ha="left",
             fontsize=12.5, color=INK, fontweight="bold")
fig.text(0.01, 0.92, "ViT width 384, 12 blocks, equal epochs; per-token values on 64 scans. Lines: median, mean of 4 seeds. "
         "Subtracting each scan's mean token first lowers these cosines by at most .016.",
         ha="left", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(OUT.with_name(OUT.stem + "_profiles.png"), dpi=160, facecolor=SURFACE)

# CKA heatmaps: one hue, light to dark
cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ee", "#9ec5ef", "#2a78d6", "#0d3a75"])
fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), facecolor=SURFACE)
for ax, (opt, label) in zip(axes, [(o, l) for o in ("Muon", "AdamW") for l in ("Plain", "Hyper-connections ×4")]):
    res = data[opt][label]
    n = len(res[0]["cka"])
    m = [[st.mean(r["cka"][i][j] for r in res) for j in range(n)] for i in range(n)]
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=1, origin="lower", extent=(0.5, n + 0.5, 0.5, n + 0.5))
    ax.set_title(f"{opt}, {label.split(' ')[0].lower() if label == 'Plain' else 'HC ×4'}", loc="left", color=INK, fontweight="bold")
    ax.set_xticks(range(1, n + 1, 2))
    ax.set_yticks(range(1, n + 1, 2))
    ax.tick_params(colors=MUTED, labelcolor=INK2, length=2)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("block", color=INK2)
axes[0].set_ylabel("block", color=INK2)
cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
cb.outline.set_visible(False)
cb.ax.tick_params(colors=MUTED, labelcolor=INK2)
cb.set_label("linear CKA", color=INK2)
fig.suptitle("Linear CKA between the 12 block outputs (readout features), mean of 4 seeds", x=0.01, y=0.99, ha="left",
             fontsize=12, color=INK, fontweight="bold")
fig.savefig(OUT.with_name(OUT.stem + "_cka.png"), dpi=160, facecolor=SURFACE, bbox_inches="tight")

# numbers for the write-up: per arm, mean over seeds
for opt in data:
    for label, res in data[opt].items():
        cs = band(res, "cos_sub")[1]
        cb_ = band(res, "cos_block")[1]
        up = band(res, "update")[1]
        n = len(res[0]["cka"])
        adj = [st.mean(r["cka"][i][i + 1] for r in res) for i in range(n - 1)]
        print(f"{opt:5s} {label[:5]:5s} cos_sub median: first-half {st.mean(cs[:11]):.3f} second-half {st.mean(cs[11:]):.3f} | "
              f"cos_block {[round(v, 3) for v in cb_]} | update attn {[round(v, 2) for v in up[0::2]]} mlp {[round(v, 2) for v in up[1::2]]} | "
              f"adjacent CKA {[round(v, 3) for v in adj]} | norm {[round(v) for v in [st.mean(r['norm'][i] for r in res) for i in range(n + 1)]]}")

# outlier channels: share of each token's energy in its largest channel, by block
RANDOM = 0.0254  # median for a random Gaussian 384-dim vector (20k samples)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=SURFACE, sharey=True)
for ax, opt in zip(axes, ("Muon", "AdamW")):
    style(ax)
    for label, color in [("Plain", PLAIN), ("Hyper-connections ×4", HC)]:
        lo, mid, hi = band(data[opt][label], "top1_share")
        x = list(range(len(mid)))
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, lw=0)
        ax.plot(x, mid, color=color, lw=2, solid_capstyle="round")
    ax.axhline(RANDOM, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(12.2, RANDOM, "random vector", color=INK2, fontsize=8, va="bottom", ha="right")
    ax.set_title(opt, loc="left", color=INK, fontweight="bold")
    ax.set_xlabel("block (0 = patch embedding; stream sum for HC)", color=INK2)
    ax.set_xticks(range(0, 13, 2))
axes[0].set_ylabel("share of token energy in largest channel", color=INK2)
axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.003, 0.925), labelcolor=INK)
fig.suptitle("Outlier channels: AdamW concentrates energy in one channel, Muon does not", x=0.01, y=0.99, ha="left",
             fontsize=12, color=INK, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.83))
fig.savefig(OUT.with_name(OUT.stem + "_outliers.png"), dpi=160, facecolor=SURFACE)
print("saved", OUT.with_name(OUT.stem + "_profiles.png"), OUT.with_name(OUT.stem + "_cka.png"), OUT.with_name(OUT.stem + "_outliers.png"))
