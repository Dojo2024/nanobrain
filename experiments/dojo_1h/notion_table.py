"""Two simple tables for Notion, written as markdown (pastes as a table) and CSV:
- notion_table.*: the 1 h experiments that helped, plus tube masking, each paired by seed against the baseline
  (last-layer readout).
- notion_layers.*: probe score at every encoder layer of the plain model (seed 7338)."""

import csv
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, "/admin/home/dojo/nanobrain/src")
from nanobrain.eval.tasks import HEADLINE_METRICS as H  # noqa: E402

O = Path("/data/dojo/nanobrain/output")
HERE = Path(__file__).parent
SEEDS = [7338, 1, 2, 3]
BASE = "baseline"
T = [("fomo_task3_t1w_hippocampus", "Hippocampus"), ("fomo_task5_t1w_cortex", "Cortex"),
     ("fomo_task2_dwi_tumor", "Tumor"), ("fomo_task1_adc_lesion", "Lesion"), ("fomo_task3_t1w_age", "Age"),
     ("fomo_task4_t2w_nerve", "Nerve"), ("fomo_task4_t2w_vessel", "Vessel"), ("fomo_task5_t1w_pmg", "PMG")]

# (experiment, what changed, run dir, verdict)
ROWS = [
    ("Baseline", "AdamW, lr 1e-3, 40-epoch cosine schedule cut at 1 h", BASE, "baseline"),
    ("lr 2e-3, 1 h decay", "AdamW lr 2e-3, schedule fitted to 1 h", "n8/lr2e-3", "positive"),
    ("lr 3e-3, 1 h decay", "AdamW lr 3e-3, schedule fitted to 1 h", "n8/lr3e-3", "positive"),
    ("lr 3e-3", "AdamW lr 3e-3, default schedule", "n8/lr3e-3-cut40", "adopted"),
    ("Tube masking", "lr 3e-3 + axial tube masking", "b2/tube-si", "negative"),
    ("Tube masking, lr 5e-3", "lr 5e-3 + axial tube masking", "b2/tube-si-lr5e-3", "negative"),
    ("Muon", "lr 3e-3 + Muon optimizer", "a17/muon", "adopted (biggest gain)"),
    ("Normalized targets", "Muon + per-patch normalized targets", "a3/normtarget", "adopted"),
    ("Faster Muon", "same as above, batched Muon (+16% speed)", "a17/batched-nt", "adopted"),
    ("Width 512", "same as above, encoder width 384 → 512", "n11/width512", "positive (best)"),
]


def score(run, seed, sub, task):
    p = O / run / f"s{seed}" / sub / f"{task}.json"
    return json.load(open(p))[H[task]] if p.exists() else None


def write(name, header, rows):
    with open(HERE / f"{name}.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    with open(HERE / f"{name}.md", "w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        f.writelines("| " + " | ".join(r) + " |\n" for r in rows)


rows = []
for name, change, run, verdict in ROWS:
    cells, used = [], set()
    for task, _ in T:
        pairs = [(s, x, y) for s in SEEDS
                 if (x := score(run, s, "eval", task)) is not None and (y := score(BASE, s, "eval", task)) is not None]
        used |= {s for s, _, _ in pairs}
        mean = st.mean(x for _, x, _ in pairs)
        diff = st.mean(x - y for _, x, y in pairs)
        cells.append(f"{mean:.3f}" if run == BASE else f"{mean:.3f} ({diff:+.3f})")
    rows.append([name, change, str(len(used))] + cells + [verdict])
rows.append(["40-epoch reference (~2.5 h)", "AdamW lr 1e-3, full schedule; 1 seed, earlier eval version", "1",
             "0.722", "0.735", "0.883", "0.743", "0.840", "0.014", "0.012", "0.875", "reference"])
write("notion_table", ["Experiment", "What changed", "Seeds"] + [n for _, n in T] + ["Verdict"], rows)

# every layer of the plain model (Muon + normalized targets, width 384), seed 7338
layers = []
for layer in range(1, 13):
    sub = "eval" if layer == 12 else f"readout/l{layer}"
    layers.append([str(layer)] + [f"{score('a17/batched-nt', 7338, sub, task):.3f}" for task, _ in T])
write("notion_layers", ["Layer"] + [n for _, n in T], layers)
print("wrote notion_table.{md,csv} and notion_layers.{md,csv} in", HERE)
