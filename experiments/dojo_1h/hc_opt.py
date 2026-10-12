"""HC effect with Muon vs with AdamW: 1 h and equal epochs, last layer and cat9-12."""
import json, statistics as st, sys
from pathlib import Path
sys.path.insert(0, "/admin/home/dojo/nanobrain/src")
from nanobrain.eval.tasks import HEADLINE_METRICS as H
O = Path("/data/dojo/nanobrain/output")
T = [("fomo_task3_t1w_hippocampus","hippo"),("fomo_task5_t1w_cortex","cortex"),("fomo_task2_dwi_tumor","tumor"),("fomo_task1_adc_lesion","lesion"),
     ("fomo_task3_t1w_age","age"),("fomo_task4_t2w_nerve","nerve"),("fomo_task4_t2w_vessel","vessel"),("fomo_task5_t1w_pmg","pmg")]
S = [7338, 1, 2, 3]
def v(run, s, sub, t):
    p = O / run / f"s{s}" / sub / f"{t}.json"
    return json.load(open(p))[H[t]] if p.exists() else None
ARMS = {"plain-Muon": "a17/batched-nt", "HC-Muon-1h": "l1/dhc4-w384", "HC-Muon-eq": "l1/dhc4-w384-matched",
        "plain-AdamW": "l1/adamw-nt-w384", "HC-AdamW-1h": "l1/dhc4-adamw-w384", "HC-AdamW-eq": "l1/dhc4-adamw-w384-matched"}
for sub in ["eval", "readout/cat9-12"]:
    print(f"\n### {sub}: absolute means (n seeds)")
    print("arm|" + "|".join(n for _, n in T))
    for a, r in ARMS.items():
        print(a + "|" + "|".join((lambda x: f"{st.mean(x):.3f}({len(x)})" if x else "-")([y for s in S if (y := v(r, s, sub, t)) is not None]) for t, _ in T))
    print(f"### {sub}: HC - plain, paired by seed")
    for lab, a, b in [("Muon 1h", "HC-Muon-1h", "plain-Muon"), ("Muon eq", "HC-Muon-eq", "plain-Muon"),
                      ("AdamW 1h", "HC-AdamW-1h", "plain-AdamW"), ("AdamW eq", "HC-AdamW-eq", "plain-AdamW")]:
        cells = []
        for t, _ in T:
            d = [x - y for s in S if (x := v(ARMS[a], s, sub, t)) is not None and (y := v(ARMS[b], s, sub, t)) is not None]
            cells.append(f"{st.mean(d):+.3f} {sum(z > 0 for z in d)}/{len(d)}" if d else "-")
        print(lab + "|" + "|".join(cells))
print("\n### final train loss (mean of 4)")
for a, r in ARMS.items():
    L = [json.loads(open(O / r / f"s{s}" / "log.json").readlines()[-1])["train/loss"] for s in S]
    E = [sum(1 for _ in open(O / r / f"s{s}" / "log.json")) for s in S]
    print(a, f"{st.mean(L):.4f}", E)
