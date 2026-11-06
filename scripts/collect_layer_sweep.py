#!/usr/bin/env python3
"""Collect and display layer sweep results for nanobrain models.

Usage:
    python scripts/collect_layer_sweep.py <model_rel_path> [--seed 7338]
Example:
    python scripts/collect_layer_sweep.py ps8/w384-d20
"""

import argparse
import json
from pathlib import Path
import pandas as pd

HEADLINE_METRICS = {
    "classification": "auroc",
    "classification_kfold": "auroc",
    "regression": "r",
    "segmentation": "dice",
    "segmentation_kfold": "dice",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="e.g. ps8/w384-d20")
    parser.add_argument("--seed", type=int, default=7338)
    parser.add_argument("--base-dir", type=Path, default=Path("/data/dojo/nanobrain/output/new_regime"))
    args = parser.parse_args()

    run_dir = args.base_dir / args.model / f"s{args.seed}"
    readout_dir = run_dir / "readout"

    if not readout_dir.exists():
        print(f"No readout directory found at {readout_dir}")
        return

    # Find all layer dirs
    subdirs = sorted(
        [d for d in readout_dir.iterdir() if d.is_dir()],
        key=lambda p: (0 if p.name.startswith("l") and p.name[1:].isdigit() else 1,
                       int(p.name[1:]) if p.name.startswith("l") and p.name[1:].isdigit() else p.name)
    )

    rows = []
    for d in subdirs:
        layer_name = d.name
        layer_results = {"layer": layer_name}
        for json_file in sorted(d.glob("*.json")):
            task = json_file.stem
            try:
                data = json.loads(json_file.read_text())
                task_type = data.get("type", "classification")
                metric_name = HEADLINE_METRICS.get(task_type, "auroc")
                val = data.get("metrics", {}).get(metric_name)
                if val is not None:
                    layer_results[task] = round(val, 4)
            except Exception as e:
                pass
        rows.append(layer_results)

    if not rows:
        print("No evaluation JSONs found yet.")
        return

    df = pd.DataFrame(rows).set_index("layer")
    print(f"\n================================================================================")
    print(f"Layer Sweep Results for {args.model} (Seed {args.seed})")
    print(f"================================================================================")
    print(df.to_markdown())
    print("\n")

    # Calculate overall task mean per layer
    numeric_cols = [c for c in df.columns if df[c].dtype in ("float64", "float32")]
    if numeric_cols:
        means = df[numeric_cols].mean(axis=1).round(4)
        print("Mean Score Across Completed Tasks:")
        for layer, m in means.items():
            print(f"  {layer:10s}: {m:.4f}")

if __name__ == "__main__":
    main()
