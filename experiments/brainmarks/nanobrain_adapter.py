"""brainmarks-smri adapter for nanobrain checkpoints. Importing this module registers model "nanobrain".

Preprocessing matches nanobrain's own eval (`nanobrain.eval.prepare`), which matches pretraining: a
threshold head mask on the native image, then `conform_image(fixed_grid=True)` to RAS, 1 mm and the
(192, 240, 192) grid (x/y centred on the head, z anchored at the top of the head), the mask
resampled nearest, and intensities clipped to the 0.5-99.5% in-mask quantiles and scaled to uint16
with zeros outside. The model z-scores inside the mask itself (`dense_embed_native`). Brainmarks'
`brain_mask` and `mni_affine` are not used: nanobrain never saw skull-stripped or MNI-aligned input.

The checkpoint can be a pretraining `checkpoint-last.pth` or a readout `model.pth` from
`nanobrain.readout`; both load through `nanobrain.eval.run.load_model`. Global tasks call the
model's `global_embed` (so readouts with their own global layers are respected), dense tasks call
`dense_embed_native` once (the model's own patch grid, which brainmarks places through
`dense_affine` and `patch_size`, not the 8 mm grid the frozen nanobrain eval needs) and take the
global embedding as the in-mask patch mean.

usage: see experiments/brainmarks/run.py
"""

from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from nibabel.processing import resample_from_to

from brainmarks_smri.models.base import EmbeddingOutput, ImageInput, register_model
from nanobrain.eval import preprocessing as preproc
from nanobrain.eval.prepare import GRID_FOV, VMAX_QUANTILE, VMIN_QUANTILE, truncate_image
from nanobrain.eval.run import load_model
from nanobrain.model import patchify3d

# brain_mode (P20, nanobrain-ideas.md 3c), using brainmarks' brain_mask:
#   none: the head mask as in nanobrain's own eval
#   pool: input unchanged; the global embedding is the mean of patches with any brain voxel
#     (from dense_embed_native, so a readout's separate global layers are not used)
#   strip: skull-stripped input (zero outside the brain, z-scored inside it); tokens, the
#     global mean and the dense mask are the brain patches
BRAIN_MODES = ("none", "pool", "strip")


class Nanobrain(nn.Module):
    name = "nanobrain"

    def __init__(self, model: nn.Module, brain_mode: str = "none"):
        super().__init__()
        assert brain_mode in BRAIN_MODES, brain_mode
        self.model = model
        self.brain_mode = brain_mode
        # a readout concatenates its layers; plain ViTMAE3D has no readout_layers
        num_layers = len(getattr(model, "readout_layers", (12,)))
        concat = getattr(model, "readout_combine", "cat") == "cat"
        self.embed_dim = model.embed_dim * (num_layers if concat else 1)
        self.patch_size = 3 * (model.patch_size,)
        self.grid_size = tuple(size // model.patch_size for size in model.grid_size)
        self.requires_grad_(False)
        self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def transform(self, sample: ImageInput) -> dict[str, Any]:
        image: nib.Nifti1Image = sample["image"]
        data = np.nan_to_num(np.asarray(image.dataobj, dtype=np.float32))
        image = nib.funcs.squeeze_image(nib.Nifti1Image(data, image.affine))

        mask_img = preproc.threshold_mask(image)
        fit_img = preproc.conform_image(image, mask_img, max_fov=GRID_FOV, fixed_grid=True)
        fit_mask = np.asarray(resample_from_to(mask_img, fit_img, order=0).dataobj) > 0
        fit_mask_img = nib.Nifti1Image(fit_mask.astype(np.uint8), fit_img.affine)
        fit_img = truncate_image(fit_img, fit_mask_img, (VMIN_QUANTILE, VMAX_QUANTILE))
        output = {
            "image": torch.from_numpy(np.asarray(fit_img.dataobj, dtype=np.float32)),
            "mask": torch.from_numpy(fit_mask),
            # model input voxel -> image world mm
            "input_affine": fit_img.affine,
        }
        if self.brain_mode != "none":
            # brainmarks' brain mask, on the image grid, resampled nearest like the head mask
            brain = np.asarray(resample_from_to(sample["brain_mask"], fit_img, order=0).dataobj) > 0
            output["brain"] = torch.from_numpy(brain & fit_mask)
        return output

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict[str, Any]], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        device = self.device
        images = torch.stack([sample["image"] for sample in samples]).to(device)
        masks = torch.stack([sample["mask"] for sample in samples]).to(device).float()
        if self.brain_mode != "none":
            brains = torch.stack([sample["brain"] for sample in samples]).to(device).float()
            assert (brains.flatten(1).sum(1) > 0).all(), "empty brain mask"
        if self.brain_mode == "strip":
            # skull-stripped input: zero outside the brain, and the model z-scores inside it
            images, masks = images * brains, brains
        # live patches: any mask voxel, as in ViTMAE3D.dense_embed_native
        kept = patchify3d(masks > 0, self.model.patch_size).any(dim=2).reshape(-1, *self.grid_size)
        pooled = kept
        if self.brain_mode == "pool":
            pooled = patchify3d(brains > 0, self.model.patch_size).any(dim=2).reshape(-1, *self.grid_size)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if return_dense or self.brain_mode == "pool":
                dense = self.model.dense_embed_native(images, masks).float()  # (B, X, Y, Z, D)
                global_embeds = [embeds[keep].mean(0) for embeds, keep in zip(dense, pooled)]
            else:
                global_embeds = list(self.model.global_embed(images, masks).float())

        outputs = []
        for ii, sample in enumerate(samples):
            output = EmbeddingOutput(
                global_embedding=global_embeds[ii],
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                output["dense_embedding"] = dense[ii]
                output["dense_mask"] = kept[ii]
                output["dense_affine"] = sample["input_affine"]
            outputs.append(output)
        return outputs


@register_model
def nanobrain(checkpoint: str, brain_mode: str = "none") -> Nanobrain:
    """A nanobrain pretraining checkpoint or readout model.pth. brain_mode (P20, see BRAIN_MODES)."""
    return Nanobrain(load_model(checkpoint), brain_mode)
