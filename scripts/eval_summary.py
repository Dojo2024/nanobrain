import json, glob, os
from pathlib import Path

# The 19 benchmark tasks and their primary metrics:
tasks_19 = [
    # Seg Core-5
    ("brats_whole_tumor", "dice"),
    ("brats_tumor_core", "dice"),
    ("brats_enhancing_tumor", "dice"),
    ("atlas_lesion", "dice"),
    ("soop_lesion", "dice"),
    # Fine Seg / Structure
    ("fomo26_task4_nerve", "dice"),
    ("fomo26_task4_vessel", "dice"),
    # Age-2
    ("ixi_age", "r2"),
    ("openbhb_age", "r2"),
    # Clinical / Diagnostic (10 tasks)
    ("fomo26_task5_pmg", "auroc"),
    ("abide_diagnosis", "auroc"),
    ("adhd_diagnosis", "auroc"),
    ("cnp_diagnosis", "auroc"),
    ("oasis1_dementia", "auroc"),
    ("oasis2_dementia", "auroc"),
    ("oasis2_conversion", "auroc"),
    ("soop_mrs_poor", "auroc"),
    ("ucsf_idh", "auroc"),
    ("upenn_survival_1y", "auroc"),
]

core5_seg = ["brats_whole_tumor", "brats_tumor_core", "brats_enhancing_tumor", "atlas_lesion", "soop_lesion"]
age2 = ["ixi_age", "openbhb_age"]

base = Path("/data/dojo/nanobrain/output/isoflop/1.0e")
runs = sorted([p for p in base.iterdir() if p.is_dir()])

import json, glob, os
from pathlib import Path

# The 19 benchmark tasks and their primary metrics:
tasks_19 = [
    # Seg Core-5
    ("brats_whole_tumor", "dice"),
    ("brats_tumor_core", "dice"),
    ("brats_enhancing_tumor", "dice"),
    ("atlas_lesion", "dice"),
    ("soop_lesion", "dice"),
    # Fine Seg / Structure
    ("fomo26_task4_nerve", "dice"),
    ("fomo26_task4_vessel", "dice"),
    # Age-2
    ("ixi_age", "r2"),
    ("openbhb_age", "r2"),
    # Clinical / Diagnostic (10 tasks)
    ("fomo26_task5_pmg", "auroc"),
    ("abide_diagnosis", "auroc"),
    ("adhd_diagnosis", "auroc"),
    ("cnp_diagnosis", "auroc"),
    ("oasis1_dementia", "auroc"),
    ("oasis2_dementia", "auroc"),
    ("oasis2_conversion", "auroc"),
    ("soop_mrs_poor", "auroc"),
    ("ucsf_idh", "auroc"),
    ("upenn_survival_1y", "auroc"),
]

core5_seg = ["brats_whole_tumor", "brats_tumor_core", "brats_enhancing_tumor", "atlas_lesion", "soop_lesion"]
age2 = ["ixi_age", "openbhb_age"]

base = Path("/data/dojo/nanobrain/output/isoflop/1.0e")
runs = sorted([p for p in base.iterdir() if p.is_dir()])

last_layer_rows = []
multi_layer_rows = []

for r in runs:
    bm_dirs = list(r.glob("s*/brainmarks"))
    for bm_dir in bm_dirs:
        res = {}
        is_multi = False
        for t, m in tasks_19:
            f = bm_dir / f"{t}.json"
            if f.exists():
                try:
                    with open(f) as fp:
                        d = json.load(fp)
                    res[t] = d.get("metrics", {}).get(m, None)
                    rl = d.get("config", {}).get("model_kwargs", {}).get("readout_layers")
                    if rl:
                        is_multi = True
                except Exception:
                    pass
        cnt = sum(1 for v in res.values() if v is not None)
        if cnt < 3:
            continue
        all19 = sum(res[t] for t, _ in tasks_19 if res.get(t) is not None) / 19.0 if cnt == 19 else None
        c5 = sum(res[t] for t in core5_seg if res.get(t) is not None) / len(core5_seg) if all(res.get(t) is not None for t in core5_seg) else None
        a2 = sum(res[t] for t in age2 if res.get(t) is not None) / len(age2) if all(res.get(t) is not None for t in age2) else None
        entry = (r.name, all19, c5, a2, res.get("ixi_age"), res.get("soop_lesion"), res.get("fomo26_task4_nerve"), res.get("fomo26_task4_vessel"), cnt)
        if is_multi:
            multi_layer_rows.append(entry)
        else:
            last_layer_rows.append(entry)

def print_table(title, rows):
    print(f"\n### {title} (N={len(rows)})")
    rows.sort(key=lambda x: (x[1] if x[1] is not None else 0, x[2] if x[2] is not None else 0), reverse=True)
    header = f"| {'Run Name':<34} | {'All-19':<7} | {'Core-5':<7} | {'Age-2':<7} | {'IXI R2':<7} | {'SOOP':<7} | {'Nerve':<7} | {'Vessel':<7} | {'Done':<5} |"
    div = "|" + "-"*36 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*9 + "|" + "-"*7 + "|"
    print(header)
    print(div)
    for name, all19, c5, a2, ixi, soop, nerve, vessel, cnt in rows:
        all19_s = f"{all19:.4f}" if all19 is not None else f"({cnt}/19)"
        c5_s = f"{c5:.4f}" if c5 is not None else "-"
        a2_s = f"{a2:.4f}" if a2 is not None else "-"
        ixi_s = f"{ixi:.4f}" if ixi is not None else "-"
        soop_s = f"{soop:.4f}" if soop is not None else "-"
        nerve_s = f"{nerve:.4f}" if nerve is not None else "-"
        vessel_s = f"{vessel:.4f}" if vessel is not None else "-"
        print(f"| {name:<34} | {all19_s:<7} | {c5_s:<7} | {a2_s:<7} | {ixi_s:<7} | {soop_s:<7} | {nerve_s:<7} | {vessel_s:<7} | {cnt:<5} |")

print_table("OFFICIAL BENCHMARK: LAST LAYER ONLY (Strict Rule)", last_layer_rows)

