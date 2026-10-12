from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

GRID_SHAPE = (192, 240, 192)
TARGET_SPACING = 1.0


class BrainNpzDataset(Dataset):
    def __init__(self, root: str | Path, filelist: str | Path):
        self.root = Path(root)
        self.paths = Path(filelist).read_text().strip().splitlines()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        with np.load(self.root / path) as npz:
            affine = npz["affine"]
            sample = {
                "path": path,
                "values": torch.from_numpy(npz["values"]),
                "mask": torch.from_numpy(npz["mask"]),
                "shape": tuple(npz["shape"].tolist()),
                "affine": torch.from_numpy(affine),
                "spacing": tuple(np.linalg.norm(affine[:3, :3], axis=0).tolist()),
            }
        return sample


def process_sample(sample: dict[str, torch.Tensor | Any], intensity_aug: bool = False):
    values = sample["values"]
    mask_rle = sample["mask"]
    shape = sample["shape"]
    spacing = sample["spacing"]

    if intensity_aug:
        mask = brle_to_dense(mask_rle, shape=shape)
        image = augment_intensity(values, mask)
    else:
        # normalize values
        values = values.float()
        values = values / values.max()
        values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)

        # decode mask
        mask = brle_to_dense(mask_rle, shape=shape)

        # make dense image
        image = torch.zeros(mask.shape, dtype=values.dtype, device=values.device)
        image.masked_scatter_(mask, values)

    # resizing
    image = resample_image(image, spacing=spacing, target_spacing=TARGET_SPACING)
    image = fit_to_shape(image, target_shape=GRID_SHAPE)

    mask = mask.float()
    mask = resample_image(mask, spacing=spacing, target_spacing=TARGET_SPACING, mode="nearest")
    mask = fit_to_shape(mask, target_shape=GRID_SHAPE)

    # apply mask
    image = image * mask

    sample = {"image": image, "mask": mask}
    return sample


# Intensity augmentation (A18, SynthSeg-style), training only: a random gamma curve and a smooth
# multiplicative bias field, applied before the in-mask z-score so the input and the target see
# the same distorted contrast. log gamma ~ U(-0.4, 0.4). The log field is a 4^3 Gaussian grid with
# a per-image std ~ U(0, 0.3), upsampled to the native image.
GAMMA_LOG_RANGE = 0.4
BIAS_GRID = 4
BIAS_MAX_STD = 0.3


def augment_intensity(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device = values.device
    values = values.float()
    values = (values / values.max()).clamp_min(0)
    log_gamma = (2 * torch.rand((), device=device) - 1) * GAMMA_LOG_RANGE
    values = values.pow(log_gamma.exp())

    image = torch.zeros(mask.shape, dtype=values.dtype, device=device)
    image.masked_scatter_(mask, values)

    field_std = torch.rand((), device=device) * BIAS_MAX_STD
    log_field = torch.randn((1, 1, BIAS_GRID, BIAS_GRID, BIAS_GRID), device=device) * field_std
    field = F.interpolate(log_field, size=mask.shape, mode="trilinear", align_corners=True)
    image = image * field[0, 0].exp()

    # in-mask z-score from masked sums, which avoids boolean indexing and its host sync
    weight = mask.float()
    count = weight.sum()
    mean = (image * weight).sum() / count
    std = (((image - mean) * weight).square().sum() / count).sqrt().clamp_min(1e-6)
    return (image - mean) / std * weight


def brle_to_dense(mask_rle: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    # mask_rle is a binary run length encoding, alternating false and true runs starting
    # with false
    X, Y, Z = shape
    run_lengths = mask_rle.long()
    run_flags = torch.arange(len(run_lengths), device=mask_rle.device) % 2 == 1
    mask = torch.repeat_interleave(run_flags, run_lengths, output_size=X * Y * Z)
    return mask.reshape(shape)


def resample_image(
    image: torch.Tensor,
    spacing: tuple[float, float, float],
    target_spacing: float = 1.0,
    mode: str = "trilinear",
) -> torch.Tensor:
    new_shape = [round(size * sp / target_spacing) for size, sp in zip(image.shape, spacing)]
    if list(image.shape) == new_shape:
        return image
    image = F.interpolate(image[None, None], size=new_shape, mode=mode)
    return image[0, 0]


def fit_to_shape(
    image: torch.Tensor, target_shape: tuple[int, int, int] = GRID_SHAPE
) -> torch.Tensor:
    # x/y centered and z top aligned, matching the crop in conform_image
    # F.pad takes pads last dim first, negative pads crop
    pad = []
    for axis in reversed(range(3)):
        diff = target_shape[axis] - image.shape[axis]
        if axis < 2:
            before = diff // 2
        else:
            before = diff
        pad += [before, diff - before]
    return F.pad(image, pad)
