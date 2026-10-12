"""Patch-size smoke helpers, run inside smoke.sbatch.

count: in-head patches per training image at patch sizes 8/12/16, counted as sample_sequences does
    (>= 25% of the patch's voxels in the head mask), to scale num_visible/num_predict so each patch
    size keeps the same fraction of the head visible and predicted.
init: a random-init checkpoint in the training format, for the eval shape checks.

usage: smoke.py count <n images>
       smoke.py init <out .pth> [overrides...]
"""

import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from nanobrain.data import BrainNpzDataset, process_sample
from nanobrain.model import ViTMAE3D, patchify3d

DATA_ROOT = "/data/nanobrain/FOMO300K_images"
DEFAULT_CONFIG = Path(__file__).parents[3] / "src/nanobrain/config/default_pretrain.yaml"


def count(n_images: int):
    dataset = BrainNpzDataset(DATA_ROOT, f"{DATA_ROOT}/filelist_subsampled.txt")
    ids = random.Random(0).sample(range(len(dataset)), n_images)
    counts = {p: [] for p in (8, 12, 16)}
    for ii in ids:
        sample = {k: v.cuda() if torch.is_tensor(v) else v for k, v in dataset[ii].items()}
        mask = process_sample(sample)["mask"][None]
        for p in counts:
            counts[p].append(int((patchify3d(mask, p).mean(dim=2) >= 0.25).sum()))
    ref = torch.tensor(counts[8], dtype=torch.float).mean()
    for p, values in counts.items():
        values = torch.tensor(values, dtype=torch.float)
        print(
            f"patch {p}: in-head patches mean {values.mean():.0f}, p5 {values.quantile(0.05):.0f}, "
            f"min {values.min():.0f}; num_visible scaled from 1280: {1280 * values.mean() / ref:.0f}"
        )


def init(out_path: str, overrides: list[str]):
    args = OmegaConf.unsafe_merge(OmegaConf.load(DEFAULT_CONFIG), OmegaConf.from_dotlist(overrides))
    args = OmegaConf.to_container(args)
    model = ViTMAE3D.from_config(args)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "args": args, "model_class": "nanobrain.model:ViTMAE3D", "git_sha": None},
        out_path,
    )
    print(f"saved {out_path}: patch {args['patch_size']}, {sum(p.numel() for p in model.parameters())} params")


if __name__ == "__main__":
    if sys.argv[1] == "count":
        count(int(sys.argv[2]))
    else:
        init(sys.argv[2], sys.argv[3:])
