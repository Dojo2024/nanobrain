"""Notion tables of every experiment (markdown pastes as a table, plus CSV), last-layer readout:
- notion_all.*: every 1 h run (HC family left out), each cell "score (diff)" with the diff paired by seed
  against the run it was built on ("Compared with"), so negatives show up.
- notion_40ep.*: the 40-epoch runs (2 seeds), paired against the adopted recipe at 40 epochs."""

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
T = [("fomo_task3_t1w_hippocampus", "Hippocampus"), ("fomo_task5_t1w_cortex", "Cortex"),
     ("fomo_task2_dwi_tumor", "Tumor"), ("fomo_task1_adc_lesion", "Lesion"), ("fomo_task3_t1w_age", "Age"),
     ("fomo_task4_t2w_nerve", "Nerve"), ("fomo_task4_t2w_vessel", "Vessel"), ("fomo_task5_t1w_pmg", "PMG")]
EXCLUDE = {"n16/ema": {2}}  # seed 2 got only 6 epochs
NAMES = {"baseline": "Baseline", "n8/lr3e-3-cut40": "lr 3e-3", "a17/muon": "Muon", "a3/normtarget": "Normalized targets",
         "a17/batched-nt": "Faster Muon", "n11/width512": "Width 512", "n11/width768": "Width 768",
         "n20/vis1024-w768": "Width 768 + 1024 visible", "n11/width512": "Width 512", "e40/adopted-w512": "Adopted recipe, 40 epochs"}

# (group, experiment, what changed, run, compared-with run, verdict); a run is "dir" or "dir:eval subdir"
ROWS = [
    ("Learning rate & schedule", "Baseline", "AdamW, lr 1e-3, 40-epoch cosine schedule cut at 1 h", "baseline", None, "baseline"),
    ("Learning rate & schedule", "lr 2e-3, 1 h decay", "lr 2e-3, schedule fitted to 1 h", "n8/lr2e-3", "baseline", "positive"),
    ("Learning rate & schedule", "lr 3e-3, 1 h decay", "lr 3e-3, schedule fitted to 1 h", "n8/lr3e-3", "baseline", "positive"),
    ("Learning rate & schedule", "lr 5e-3, 1 h decay", "lr 5e-3, schedule fitted to 1 h", "n8/lr5e-3", "baseline", "negative"),
    ("Learning rate & schedule", "lr 8e-3, 1 h decay", "lr 8e-3, schedule fitted to 1 h", "n8/lr8e-3", "baseline", "negative (unstable)"),
    ("Learning rate & schedule", "Cosine over 14 epochs", "schedule fitted to 1 h, 1 warmup epoch", "n7/cos14", "baseline", "negative"),
    ("Learning rate & schedule", "Linear decay over 14 epochs", "linear decay to 0 at 1 h", "n7/lin14", "baseline", "negative"),
    ("Learning rate & schedule", "Time-keyed decay", "lr decays with elapsed wall time", "n7/time", "baseline", "negative"),
    ("Learning rate & schedule", "lr 3e-3", "lr 3e-3, default schedule", "n8/lr3e-3-cut40", "baseline", "adopted"),
    ("Learning rate & schedule", "lr 4e-3", "lr 4e-3, default schedule", "n8/lr4e-3-cut40", "n8/lr3e-3-cut40", "inconclusive (1 seed)"),
    ("Learning rate & schedule", "lr 5e-3", "lr 5e-3, default schedule", "n8/lr5e-3-cut40", "n8/lr3e-3-cut40", "negative"),
    ("Masking", "Tube masking", "axial tube masking", "b2/tube-si", "n8/lr3e-3-cut40", "negative (lesion up)"),
    ("Masking", "Tube masking, lr 5e-3", "axial tube masking, lr 5e-3", "b2/tube-si-lr5e-3", "n8/lr3e-3-cut40", "negative (lesion up)"),
    ("Masking", "32 mm block masking", "mask 4×4×4-patch cubes", "b1/block4", "n8/lr3e-3-cut40", "negative"),
    ("Optimizer", "Muon", "Muon optimizer", "a17/muon", "n8/lr3e-3-cut40", "adopted (biggest gain)"),
    ("Optimizer", "Muon lr ×2", "Muon learning rate doubled", "a17/muon-lr2x", "a17/muon", "negative"),
    ("Optimizer", "AdamW + normalized targets", "AdamW instead of Muon, width 384", "l1/adamw-nt-w384", "a17/batched-nt", "mixed (lesion up, rest down)"),
    ("Optimizer", "Muon lr ×0.5, width 512", "Muon learning rate halved", "n8/muonlr05-w512", "n11/width512", "negative"),
    ("Targets & decoder", "Normalized targets", "per-patch normalized targets", "a3/normtarget", "a17/muon", "adopted"),
    ("Targets & decoder", "Faster Muon", "batched Muon (+16% speed)", "a17/batched-nt", "a3/normtarget", "adopted"),
    ("Targets & decoder", "Stats head", "decoder also predicts each patch's mean and std", "n17/statshead", "a17/batched-nt", "mixed"),
    ("Targets & decoder", "Decoder 10×192", "deeper, narrower decoder", "n18/dec10x192", "a17/batched-nt", "negative"),
    ("Targets & decoder", "Cross-attention decoder", "decoder of cross-attention only, width 512", "n19/xdec", "n11/width512", "mixed (anatomy up, lesion down)"),
    ("Sampling & averaging", "Balanced modality sampling", "sqrt-balanced contrasts (slowed loader)", "b26/sqrt", "a17/muon", "confounded"),
    ("Sampling & averaging", "Balanced sampling, rerun", "same, clean rerun", "b26/sqrt-r2", "a17/muon", "null"),
    ("Sampling & averaging", "EMA of weights 0.999", "weight average, same run (3 seeds)", "n16/ema:eval-ema0.999", "n16/ema", "null"),
    ("Model size & masking budget", "Width 512", "encoder width 384 → 512", "n11/width512", "a17/batched-nt", "adopted"),
    ("Model size & masking budget", "Width 640", "encoder width 640", "n11/width640", "n11/width512", "mixed (anatomy up, tumor/age down)"),
    ("Model size & masking budget", "Width 768", "encoder width 768", "n11/width768", "n11/width512", "mixed (anatomy up, tumor/age down)"),
    ("Model size & masking budget", "1024 visible, width 512", "1024 instead of 1280 visible patches", "n20/vis1024-w512", "n11/width512", "null"),
    ("Model size & masking budget", "1024 visible, width 768", "1024 instead of 1280 visible patches", "n20/vis1024-w768", "n11/width768", "positive (best 1 h model)"),
    ("Model size & masking budget", "Short warmup", "2 warmup epochs instead of 5", "n21/wu2-vis1024-w768", "n20/vis1024-w768", "negative"),
    ("Model size & masking budget", "One mask per image", "64 images × 1 mask instead of 32 × 2", "n9/onemask-w512", "n11/width512", "cancelled (slower steps)"),
    ("Augmentation & tokens", "Intensity augmentation", "random gamma + bias field", "n22/intaug-w512", "n11/width512", "negative (lesion unchanged)"),
    ("Augmentation & tokens", "Register tokens", "4 extra learnable tokens", "n22/reg4-w512", "n11/width512", "mixed (lesion up)"),
    ("Augmentation & tokens", "Register tokens on best", "4 extra tokens, width 768 + 1024 visible", "n22/reg4-vis1024-w768", "n20/vis1024-w768", "negative (tumor down)"),
    ("Model size & masking budget", "Width 768, depth 8", "8 encoder layers instead of 12", "n23/w768-d8", "n11/width512", "positive (best 5-task mean)"),
]


def split(run):
    d, _, sub = run.partition(":")
    return d, sub or "eval"


def score(run, seed, task):
    d, sub = split(run)
    p = O / d / f"s{seed}" / sub / f"{task}.json"
    return json.load(open(p))[H[task]] if p.exists() and seed not in EXCLUDE.get(d, ()) else None


def epochs(run):
    d, _ = split(run)
    n = [sum(1 for line in open(p) if line.strip()) for s in SEEDS if (p := O / d / f"s{s}" / "log.json").exists()
         and s not in EXCLUDE.get(d, ())]
    return f"{st.mean(n):.0f}" if n else "–"


def cells(run, base):
    out, used = [], set()
    for task, _ in T:
        if base is None:
            xs = [x for s in SEEDS if (x := score(run, s, task)) is not None]
            used |= {s for s in SEEDS if score(run, s, task) is not None}
            out.append(f"{st.mean(xs):.3f}")
            continue
        pairs = [(x, y) for s in SEEDS if (x := score(run, s, task)) is not None and (y := score(base, s, task)) is not None]
        used |= {s for s in SEEDS if score(run, s, task) is not None and score(base, s, task) is not None}
        out.append(f"{st.mean(x for x, _ in pairs):.3f} ({st.mean(x - y for x, y in pairs):+.3f})" if pairs else "–")
    return out, len(used)


def write(name, header, rows):
    with open(HERE / f"{name}.csv", "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    with open(HERE / f"{name}.md", "w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        f.writelines("| " + " | ".join(r) + " |\n" for r in rows)


rows = []
for group, name, change, run, base, verdict in ROWS:
    c, n = cells(run, base)
    against = "–" if base is None else NAMES.get(split(base)[0], name if split(base)[0] == split(run)[0] else base)
    if run.startswith("n16/ema"):
        against = "Same run, raw weights"
    ep = "–" if verdict.startswith(("cancelled", "running")) else epochs(run)
    rows.append([group, name, change, against, str(n) if n else "–", ep] + c + [verdict])
rows.append(["Reference", "40-epoch reference (~2.5 h)", "AdamW lr 1e-3, full schedule; earlier eval version", "–", "1", "40",
             "0.722", "0.735", "0.883", "0.743", "0.840", "0.014", "0.012", "0.875", "reference"])
header = ["Group", "Experiment", "What changed", "Compared with", "Seeds", "Epochs"] + [n for _, n in T] + ["Verdict"]
write("notion_all", header, rows)

E40 = [
    ("Adopted recipe, 40 epochs", "Muon + normalized targets, width 512", "e40/adopted-w512", None, "reference"),
    ("Width 768, 40 epochs", "encoder width 768", "e40/width768", "e40/adopted-w512", "positive (best overall)"),
    ("AdamW, 40 epochs", "AdamW instead of Muon", "e40/adamw-nt-w512", "e40/adopted-w512", "mixed (lesion/cortex up, rest down)"),
    ("Tube masking, 40 epochs", "axial tube masking", "e40/tube-w512", "e40/adopted-w512", "negative (PMG up only)"),
    ("Adopted recipe, 1 h", "same recipe, 1 h (same 2 seeds)", "n11/width512", "e40/adopted-w512", "1 h vs 40 epochs"),
]
SEEDS = [7338, 1]
rows = []
for name, change, run, base, verdict in E40:
    c, n = cells(run, base)
    rows.append([name, change, "–" if base is None else NAMES[base], str(n), epochs(run)] + c + [verdict])
rows.append(["Connor's 40-epoch run", "AdamW lr 1e-3, width 384; earlier eval version", "–", "1", "40",
             "0.722", "0.735", "0.883", "0.743", "0.840", "0.014", "0.012", "0.875", "reference"])
write("notion_40ep", ["Experiment", "What changed", "Compared with", "Seeds", "Epochs"] + [n for _, n in T] + ["Verdict"], rows)
print("wrote notion_all.{md,csv} and notion_40ep.{md,csv} in", HERE)
