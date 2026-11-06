from pathlib import Path
import json

base = Path("/data/dojo/nanobrain/output/isoflop")
rows = []

for scale in ["1.0e", "2.0e", "4.0e", "8.0e", "17.6e"]:
    scale_dir = base / scale
    if not scale_dir.exists():
        continue
    for run_dir in sorted(scale_dir.glob("*/*")):
        agg_file = run_dir / "brainmarks" / "aggregate.json"
        if not agg_file.exists():
            continue
        try:
            data = json.loads(agg_file.read_text())
            s = data.get("summary", {})
            rows.append({
                "name": f"{scale}/{run_dir.parent.name}/{run_dir.name}",
                "all": s.get("all", {}).get("score"),
                "core": s.get("core", {}).get("score"),
                "open": s.get("open", {}).get("score"),
                "seg": s.get("segmentation", {}).get("score"),
                "glob": s.get("global", {}).get("score"),
                "brats_et": s.get("brats_enhancing_tumor", {}).get("score"),
                "brats_tc": s.get("brats_tumor_core", {}).get("score"),
                "atlas": s.get("atlas_lesion", {}).get("score"),
                "ixi": s.get("ixi_age", {}).get("score"),
                "adni_ad": s.get("adni_dementia", {}).get("score"),
            })
        except Exception:
            pass

rows.sort(key=lambda r: (r["core"] or 0, r["all"] or 0), reverse=True)

fmt = "{:<58} {:<8} {:<8} {:<8} {:<8} {:<8} {:<7} {:<7} {:<7} {:<7}"
print(fmt.format("Model (Scale/Run/Seed)", "Core-18", "Open-11", "All-31", "Seg-7", "Glob-24", "ET", "TC", "Atlas", "IXI"))
print("-" * 135)
for r in rows:
    def f(v):
        return f"{v:.4f}" if v is not None else "N/A"
    print(fmt.format(
        r["name"][:57],
        f(r["core"]),
        f(r["open"]),
        f(r["all"]),
        f(r["seg"]),
        f(r["glob"]),
        f(r["brats_et"]),
        f(r["brats_tc"]),
        f(r["atlas"]),
        f(r["ixi"]),
    ))

