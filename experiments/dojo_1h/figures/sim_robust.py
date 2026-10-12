"""Summarize the robustness fields of similarity.json: raw vs centered cosine, shared direction, dominant channels."""
import json, statistics as st
from pathlib import Path
O = Path("/data/dojo/nanobrain/output")
S = [7338, 1, 2, 3]
ARMS = [("Muon plain", "a17/batched-nt"), ("Muon HC", "l1/dhc4-w384-matched"),
        ("AdamW plain", "l1/adamw-nt-w384"), ("AdamW HC", "l1/dhc4-adamw-w384-matched")]
m = lambda R, key, i, q=1: st.mean(r[key][i][q] for r in R)
print("arm | check_err | cos_sub 2nd half raw / centered | cos_block b6-10 raw / centered | CKA(1,12) raw / centered | CKA(9,12) raw / centered")
for name, run in ARMS:
    R = [json.loads((O / run / f"s{s}" / "similarity.json").read_text()) for s in S]
    cs = lambda key: st.mean(m(R, key, i) for i in range(11, 23))
    cb = lambda key: st.mean(m(R, key, i) for i in range(5, 10))
    ck = lambda key, i, j: st.mean(r[key][i][j] for r in R)
    print(f"{name} | {max(r['check_err'] for r in R):.1e} | {cs('cos_sub'):.3f} / {cs('cos_sub_c'):.3f} | {cb('cos_block'):.3f} / {cb('cos_block_c'):.3f} | "
          f"{ck('cka', 0, 11):.3f} / {ck('cka_c', 0, 11):.3f} | {ck('cka', 8, 11):.3f} / {ck('cka_c', 8, 11):.3f}")
print("\narm | layer | mean-token share | top-1 channel share | top-4 share | max/median | most-often-top channel per seed (fraction)")
for name, run in ARMS:
    R = [json.loads((O / run / f"s{s}" / "similarity.json").read_text()) for s in S]
    for layer in (0, 1, 2, 6, 12):
        tc = " ".join(f"{r['top_channel'][layer][0]}({r['top_channel'][layer][1]:.2f})" for r in R)
        print(f"{name} | {layer} | {m(R, 'mean_share', layer):.3f} | {m(R, 'top1_share', layer):.3f} | {m(R, 'top4_share', layer):.3f} | {m(R, 'top_ratio', layer):.1f} | {tc}")
