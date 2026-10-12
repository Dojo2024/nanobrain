"""Plain vs HC vs mHC (Muon, width 384, equal epochs), 4 seeds: layers 3, 6, 9, 12 one at a time,
and the split readout (dense tasks from layers 6+9+12, global tasks from layers 9+12)."""
import json, statistics as st, sys
from pathlib import Path
sys.path.insert(0, "/admin/home/dojo/nanobrain/src")
from nanobrain.eval.tasks import HEADLINE_METRICS as H

O = Path("/data/dojo/nanobrain/output")
S = [7338, 1, 2, 3]
T = [("fomo_task3_t1w_hippocampus", "hippo"), ("fomo_task5_t1w_cortex", "cortex"), ("fomo_task2_dwi_tumor", "tumor"),
     ("fomo_task1_adc_lesion", "lesion"), ("fomo_task3_t1w_age", "age"), ("fomo_task4_t2w_nerve", "nerve"),
     ("fomo_task4_t2w_vessel", "vessel"), ("fomo_task5_t1w_pmg", "pmg")]
GLOBAL = {"fomo_task3_t1w_age", "fomo_task5_t1w_pmg"}
R = {"plain": "a17/batched-nt", "HC": "l1/dhc4-w384-matched", "mHC": "l1/mhc4-w384"}


def sub(ro, t):
    if ro == "l12":
        return "eval"
    if ro == "split":
        return "readout/cat9-12" if t in GLOBAL else "readout/cat6-9-12"
    return f"readout/{ro}"


def v(r, s, ro, t):
    return json.load(open(O / r / f"s{s}" / sub(ro, t) / f"{t}.json"))[H[t]]


def d(a, b, ro, t):
    x = [v(R[a], s, ro, t) - v(R[b], s, ro, t) for s in S]
    return f"{st.mean(x):+.3f} ({sum(y > 0 for y in x)}/4)"


for ro in ["l3", "l6", "l9", "l12", "split"]:
    print(f"\n## {ro}: plain | HC | mHC || HC-plain | mHC-plain | mHC-HC")
    for t, n in T:
        m = [st.mean(v(R[k], s, ro, t) for s in S) for k in R]
        print(f"{n:7s} {m[0]:.3f} | {m[1]:.3f} | {m[2]:.3f} || {d('HC', 'plain', ro, t)} | {d('mHC', 'plain', ro, t)} | {d('mHC', 'HC', ro, t)}")
