from pathlib import Path
import json

tasks_19 = [
    ("brats_whole_tumor", "dice"),
    ("brats_tumor_core", "dice"),
    ("brats_enhancing_tumor", "dice"),
    ("atlas_lesion", "dice"),
    ("soop_lesion", "dice"),
    ("fomo26_task4_nerve", "dice"),
    ("fomo26_task4_vessel", "dice"),
    ("ixi_age", "r2"),
    ("openbhb_age", "r2"),
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

base = Path("/data/dojo/nanobrain/output/isoflop/2.0e")
runs = []
for p in base.glob("*/*"):
    bm_dir = p / "brainmarks"
    if not bm_dir.exists():
        continue
    
    res = {}
    is_multi = False
    for t, m in tasks_19:
        f = bm_dir / f"{t}.json"
        if f.exists():
            try:
                d = json.loads(f.read_text())
                val = d.get("metrics", {}).get(m)
                if val is not None:
                    res[t] = val
                rl = d.get("config", {}).get("model_kwargs", {}).get("readout_layers")
                if rl:
                    is_multi = True
            except Exception:
                pass
    
    cnt = len(res)
    if cnt == 0:
        continue
    all19 = sum(res[t] for t, _ in tasks_19 if t in res) / 19.0 if cnt == 19 else None
    c5 = sum(res[t] for t in core5_seg if t in res) / 5.0 if all(t in res for t in core5_seg) else None
    a2 = sum(res[t] for t in age2 if t in res) / 2.0 if all(t in res for t in age2) else None
    
    runs.append({
        "name": f"{p.parent.name}/{p.name}",
        "cnt": cnt,
        "is_multi": is_multi,
        "all19": all19,
        "c5": c5,
        "a2": a2,
        "tc": res.get("brats_tumor_core"),
        "wt": res.get("brats_whole_tumor"),
        "et": res.get("brats_enhancing_tumor"),
        "atlas": res.get("atlas_lesion"),
        "soop": res.get("soop_lesion"),
    })

last_layer = [r for r in runs if not r["is_multi"]]
last_layer.sort(key=lambda x: (x["cnt"] == 19, x["all19"] or 0, x["c5"] or 0), reverse=True)

fmt = "{:<55} {:<5} {:<8} {:<8} {:<8} {:<7} {:<7} {:<7} {:<7}"
print(fmt.format("Run", "Tasks", "All-19", "Core-5", "Age-2", "TC", "WT", "ET", "Atlas"))
print("-" * 115)
for r in last_layer[:25]:
    all_s = "{:.4f}".format(r["all19"]) if r["all19"] is not None else "Prog({}/19)".format(r["cnt"])
    c5_s = "{:.4f}".format(r["c5"]) if r["c5"] is not None else "N/A"
    a2_s = "{:.4f}".format(r["a2"]) if r["a2"] is not None else "N/A"
    tc_s = "{:.4f}".format(r["tc"]) if r["tc"] is not None else "N/A"
    wt_s = "{:.4f}".format(r["wt"]) if r["wt"] is not None else "N/A"
    et_s = "{:.4f}".format(r["et"]) if r["et"] is not None else "N/A"
    at_s = "{:.4f}".format(r["atlas"]) if r["atlas"] is not None else "N/A"
    print(fmt.format(r["name"][:54], r["cnt"], all_s, c5_s, a2_s, tc_s, wt_s, et_s, at_s))
