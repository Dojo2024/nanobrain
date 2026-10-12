| Experiment | What changed | Compared with | Seeds | Epochs | Hippocampus | Cortex | Tumor | Lesion | Age | Nerve | Vessel | PMG | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Adopted recipe, 40 epochs | Muon + normalized targets, width 512 | – | 2 | 40 | 0.810 | 0.744 | 0.900 | 0.694 | 0.894 | 0.018 | 0.015 | 0.911 | reference |
| Width 768, 40 epochs | encoder width 768 | Adopted recipe, 40 epochs | 2 | 40 | 0.825 (+0.015) | 0.766 (+0.021) | 0.908 (+0.008) | 0.707 (+0.013) | 0.897 (+0.003) | 0.023 (+0.005) | 0.017 (+0.002) | 0.921 (+0.010) | positive (best overall) |
| AdamW, 40 epochs | AdamW instead of Muon | Adopted recipe, 40 epochs | 2 | 40 | 0.789 (-0.021) | 0.768 (+0.024) | 0.885 (-0.015) | 0.743 (+0.048) | 0.854 (-0.040) | 0.021 (+0.003) | 0.016 (+0.001) | 0.945 (+0.035) | mixed (lesion/cortex up, rest down) |
| Tube masking, 40 epochs | axial tube masking | Adopted recipe, 40 epochs | 2 | 40 | 0.805 (-0.005) | 0.731 (-0.014) | 0.899 (-0.001) | 0.687 (-0.007) | 0.894 (-0.000) | 0.021 (+0.003) | 0.014 (-0.001) | 0.963 (+0.052) | negative (PMG up only) |
| Adopted recipe, 1 h | same recipe, 1 h (same 2 seeds) | Adopted recipe, 40 epochs | 2 | 13 | 0.801 (-0.009) | 0.750 (+0.005) | 0.905 (+0.005) | 0.700 (+0.006) | 0.879 (-0.015) | 0.021 (+0.004) | 0.016 (+0.001) | 0.946 (+0.036) | 1 h vs 40 epochs |
| Connor's 40-epoch run | AdamW lr 1e-3, width 384; earlier eval version | – | 1 | 40 | 0.722 | 0.735 | 0.883 | 0.743 | 0.840 | 0.014 | 0.012 | 0.875 | reference |
