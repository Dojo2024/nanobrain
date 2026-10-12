"""Check that the adapter's input grid lines up with brainmarks segmentation labels: map each label
onto the model input grid through `input_affine`, as probes.segmentation_patches does, and report
how much of it lands inside the head mask and how many voxels survive (1 mm grid, so the count
should match the label volume in mm^3). Also checks the dense output shapes on a forward.

usage: check_alignment.py <checkpoint> [task ...]"""

import sys

import numpy as np
import torch

import nanobrain_adapter  # noqa: F401  registers model "nanobrain"
from brainmarks_smri.models import create_model
from brainmarks_smri.probes import label_mask, resample
from brainmarks_smri.tasks import create_task

checkpoint = sys.argv[1]
tasks = sys.argv[2:] or ["brats_whole_tumor", "soop_lesion"]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_model("nanobrain", checkpoint=checkpoint).to(device).eval()
print(f"model: embed_dim {model.embed_dim}, patch {model.patch_size}, grid {model.grid_size}")

for name in tasks:
    task = create_task(name, max_per_split=4)
    task.dataset.transform = model.transform
    for index in task.eval_ids[:3]:
        sample, targets = task.dataset[int(index)]
        label = targets[task.target]
        mask = label_mask(label, task.label_values, "cpu")
        mm3 = float(mask.sum()) * abs(np.linalg.det(label.affine[:3, :3]))
        input_to_label = np.linalg.inv(label.affine) @ sample["input_affine"]
        on_input = resample(mask[None].float(), input_to_label, tuple(sample["image"].shape), "nearest")[0] > 0.5
        inside = float((on_input & sample["mask"]).sum()) / max(float(on_input.sum()), 1.0)
        print(
            f"{name} {targets['participant_id']}: label {mm3:.0f} mm3 -> {int(on_input.sum())} input voxels, "
            f"{inside:.1%} inside head mask, head mask {int(sample['mask'].sum())} voxels, "
            f"image max {float(sample['image'].max()):.0f}"
        )
    (output,) = model.forward_embeddings([sample], return_dense=True)
    print(
        f"{name} forward: global {tuple(output['global_embedding'].shape)}, dense "
        f"{tuple(output['dense_embedding'].shape)}, kept {int(output['dense_mask'].sum())} patches, "
        f"finite {bool(torch.isfinite(output['dense_embedding']).all())}"
    )
