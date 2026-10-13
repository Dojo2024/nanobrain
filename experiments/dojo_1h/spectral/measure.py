"""Size of the A2 spectral term next to the masked MSE, on the same real training batches, for one or
more checkpoints. Both are means over the masked patches, as in ViTMAE3D.forward, so the spectral
term's share of the loss at spectral_weight w is w * spectral / mse.

usage: python experiments/dojo_1h/spectral/measure.py <checkpoint> [<checkpoint> ...]
"""

import sys

import torch
from torch.utils.data import DataLoader, default_collate

import nanobrain.utils.misc as misc
from nanobrain.data import BrainNpzDataset, process_sample
from nanobrain.eval.run import load_model
from nanobrain.model import spectral_loss

DATA_ROOT = "/data/nanobrain/FOMO300K_images"
NUM_BATCHES = 8

dataset = BrainNpzDataset(DATA_ROOT, f"{DATA_ROOT}/filelist_subsampled.txt")
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8, collate_fn=list,
                    generator=torch.Generator().manual_seed(0))
batches = []
for batch in loader:
    batch = default_collate([process_sample(s) for s in misc.send_data(batch)])
    batches.append((batch["image"], batch["mask"]))
    if len(batches) == NUM_BATCHES:
        break

for ckpt in sys.argv[1:]:
    args = torch.load(ckpt, map_location="cpu", weights_only=True)["args"]
    model = load_model(ckpt).cuda().eval()
    mse, spec = [], []
    torch.manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for images, masks in batches:
            _, state = model(images, masks, num_visible=args["num_visible"], num_predict=args["num_predict"],
                             num_samples=args["num_samples"], norm_target=args["norm_target"])
            pred = state["pred_patches"].float()
            target, m = state["target_patches"].float(), state["target_mask_patches"].float()
            mse.append(((pred - target) ** 2 * m).sum() / m.sum())
            per_patch = spectral_loss(pred, target, m, model.spectral_mask, model.patch_size)
            voxels = m.sum(dim=-1)
            spec.append((per_patch * voxels).sum() / voxels.sum())
    mse, spec = torch.stack(mse).mean().item(), torch.stack(spec).mean().item()
    print(f"{ckpt}\n  masked MSE {mse:.4f}  spectral {spec:.4f}  "
          f"share at w=0.01/0.1/1: {0.01 * spec / mse:.2f} / {0.1 * spec / mse:.2f} / {spec / mse:.2f}")
