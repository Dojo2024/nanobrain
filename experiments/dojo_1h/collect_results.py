"""Collect the 1 h ablation results into small CSVs.

Reads <output>/<run>/s<seed>/{config.yaml,log.json,eval/*.json,readout/*/*.json} and writes
  runs.csv     one row per run and seed: config overrides vs the default config, epochs, final loss
  results.csv  one row per run, seed, readout and task: the headline metric and its 95% CI
Only scalar metrics are kept; the per-subject targets and predictions in the eval JSONs are not.

usage: python experiments/dojo_1h/collect_results.py /data/dojo/nanobrain/output
"""

import argparse
import csv
import json
from pathlib import Path

import yaml

from nanobrain.eval.tasks import DEFAULT_TASKS, HEADLINE_METRICS

HERE = Path(__file__).parent
DEFAULT_CONFIG = HERE.parents[1] / "src/nanobrain/config/default_pretrain.yaml"
# set per run by the launch script, not ablation settings
IGNORED_KEYS = {"name", "notes", "output_dir", "seed", "data_root", "train_filelist", "lr"}
# the 2026-10-03 sweep named the 4 layer concat of layers 9-12 cat9-12; later runs use that name
# for the 2 layer concat of layers 9 and 12
RENAMED_READOUTS = {("baseline", "cat9-12"): "cat9-10-11-12"}


def same(a, b):
    # yaml loads 1e-6 as a string but the saved config writes it as a float
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def overrides(config, default):
    return {
        k: v for k, v in config.items() if k not in IGNORED_KEYS and not same(default.get(k, v), v)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    default = yaml.safe_load(DEFAULT_CONFIG.read_text())
    runs, results = [], []
    for config_path in sorted(args.output.glob("**/s*/config.yaml")):
        seed_dir = config_path.parent
        run = str(seed_dir.parent.relative_to(args.output))
        seed = int(seed_dir.name[1:])
        config = yaml.safe_load(config_path.read_text())
        log = [json.loads(line) for line in (seed_dir / "log.json").read_text().splitlines()]
        runs.append({
            "run": run,
            "seed": seed,
            "overrides": json.dumps(overrides(config, default)),
            "epochs_done": len(log),
            "final_train_loss": round(log[-1]["train/loss"], 4) if log else "",
            "mean_vol_per_s": round(sum(r["train/vol/s"] for r in log) / len(log)) if log else "",
        })

        readout_dirs = [("l12", seed_dir / "eval")]
        readout_dirs += [(d.name, d) for d in sorted((seed_dir / "readout").glob("*")) if d.is_dir()]
        # EMA checkpoints (ema_decays) are scored on the last layer, eval-ema<decay> -> ema<decay>-l12
        readout_dirs += [(f"{d.name[len('eval-'):]}-l12", d) for d in sorted(seed_dir.glob("eval-ema*"))]
        for readout, result_dir in readout_dirs:
            readout = RENAMED_READOUTS.get((run, readout), readout)
            for task in DEFAULT_TASKS:
                path = result_dir / f"{task}.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                metric = HEADLINE_METRICS[task]
                ci = result.get(f"{metric}_ci", [None, None])
                results.append({
                    "run": run,
                    "seed": seed,
                    "readout": readout,
                    "task": task,
                    "metric": metric,
                    "value": round(result[metric], 5),
                    "ci_low": round(ci[0], 5) if ci[0] is not None else "",
                    "ci_high": round(ci[1], 5) if ci[1] is not None else "",
                    "embed_dim": result.get("embed_dim", ""),
                })

    for name, rows in [("runs.csv", runs), ("results.csv", results)]:
        with (HERE / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
