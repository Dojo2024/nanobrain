"""Paired comparison of an arm against its control, seed by seed, on FOMO and brainmarks-smri.

FOMO (last layer = eval/, plus every readout both runs have): arm mean, mean paired difference, and how
many seeds improved.
brainmarks: arm mean, mean paired difference, and [k/n] = seeds whose paired bootstrap 95% CI over eval
subjects (1000 resamples) excludes 0, net of direction (+ better, - worse). The bootstrap only captures
test-set noise, not seed noise, so read it together with the seed count.

usage: PYTHONPATH=~/brainmarks-smri/src python experiments/dojo_1h/paired.py <arm> <control>
           [--seeds 7338 1 2 3] [--variant brainmarks] [--control-variant brainmarks]
  e.g. width 1024:  paired.py ps16/w1024-sub2 ps16/w768-sub2
       P20 pool:    paired.py ps16/w768-sub2 ps16/w768-sub2 --seeds 7338 1 --variant brainmarks-brainpool
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from brainmarks_smri.metrics import auroc
from nanobrain.eval.tasks import DEFAULT_TASKS, HEADLINE_METRICS

OUT = Path("/data/dojo/nanobrain/output")
BM_TASKS = ["abide_diagnosis", "cnp_diagnosis", "ucsf_idh", "upenn_survival_1y", "soop_mrs_poor", "ixi_age",
            "openbhb_age", "brats_whole_tumor", "brats_tumor_core", "brats_enhancing_tumor", "soop_lesion"]


def fomo(path, task):
    try:
        return json.load(open(path / f"{task}.json"))[HEADLINE_METRICS[task]]
    except FileNotFoundError:
        return None


def bm_scorer(path):
    # (subject keys, score function over subject indices) from the per-subject eval_samples
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        return None, None
    rows = {(s["participant_id"], s["session_id"]): s for s in d["eval_samples"]}
    keys = sorted(rows)
    s0 = rows[keys[0]]
    if d["type"] == "classification":
        t = next(k for k in s0 if k not in ("participant_id", "session_id", "probabilities"))
        y = np.array([d["classes"].index(rows[k][t]) for k in keys])
        # probability keys are the JSON spelling of the class ("true", not True)
        p = np.array([[rows[k]["probabilities"][json.dumps(c).strip('"')] for c in d["classes"]] for k in keys])
        return keys, lambda ids: auroc(y[ids], p[ids])
    if d["type"] == "regression":
        t = next(k for k in s0 if k not in ("participant_id", "session_id", "prediction"))
        y = np.array([rows[k][t] for k in keys])
        p = np.array([rows[k]["prediction"] for k in keys])
        return keys, lambda ids: r2_score(y[ids], p[ids])
    v = np.array([rows[k]["dice"] for k in keys])
    return keys, lambda ids: np.nanmean(v[ids])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("control")
    ap.add_argument("--seeds", nargs="+")
    ap.add_argument("--variant", default="brainmarks")
    ap.add_argument("--control-variant", default="brainmarks")
    args = ap.parse_args()
    arm, ctrl = OUT / args.arm, OUT / args.control
    seeds = args.seeds or sorted({p.name[1:] for p in arm.glob("s*")} & {p.name[1:] for p in ctrl.glob("s*")})
    print(f"{args.arm} vs {args.control}, seeds {' '.join(seeds)}")

    if args.arm != args.control:
        readouts = ["eval"] + sorted({p.name for p in arm.glob("s*/readout/*") if p.is_dir()}
                                     & {p.name for p in ctrl.glob("s*/readout/*") if p.is_dir()})
        for ro in readouts:
            print(f"\nFOMO {ro if ro != 'eval' else 'last layer'}")
            for t in DEFAULT_TASKS:
                sub = ro if ro == "eval" else f"readout/{ro}"
                pairs = [(fomo(arm / f"s{s}" / sub, t), fomo(ctrl / f"s{s}" / sub, t)) for s in seeds]
                pairs = [(a, c) for a, c in pairs if a is not None and c is not None]
                if not pairs:
                    print(f"  {t:28} -")
                    continue
                d = [a - c for a, c in pairs]
                print(f"  {t:28} {np.mean([a for a, _ in pairs]):.3f}  {np.mean(d):+.3f} "
                      f"({sum(x > 0 for x in d)}/{len(d)})")

    print(f"\nbrainmarks {args.variant} vs {args.control_variant}")
    rng = np.random.default_rng(0)
    for t in BM_TASKS:
        vals, diffs, sig = [], [], 0
        for s in seeds:
            keys, f = bm_scorer(arm / f"s{s}" / args.variant / f"{t}.json")
            ckeys, cf = bm_scorer(ctrl / f"s{s}" / args.control_variant / f"{t}.json")
            if f is None or cf is None:
                continue
            assert keys == ckeys, (t, s, "different eval subjects")
            n = len(keys)
            boot = np.array([f(ids) - cf(ids) for ids in (rng.integers(0, n, n) for _ in range(1000))])
            lo, hi = np.percentile(boot[~np.isnan(boot)], [2.5, 97.5])
            vals.append(f(np.arange(n)))
            diffs.append(vals[-1] - cf(np.arange(n)))
            sig += int(lo > 0) - int(hi < 0)
        if diffs:
            print(f"  {t:24} {np.mean(vals):.3f}  {np.mean(diffs):+.3f} [{sig:+d}/{len(diffs)}]")
        else:
            print(f"  {t:24} -")


if __name__ == "__main__":
    main()
