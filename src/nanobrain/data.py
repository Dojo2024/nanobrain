import itertools
import math
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


def process_batch(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Batched version of process_batch_per_image, bitwise identical: the per image max / mean /
    std keep their own reductions (torch's reduction order depends on the tensor, so they can't
    be batched bitwise), everything else runs once on the concatenated batch: the cast and
    z-scoring, the run length decode (one cumsum over the concatenated runs gives the flat voxel
    index into one concatenated dense buffer), the scatters, and the zeroing of the batch. About
    300 kernel launches per batch of 32 instead of ~1100 (the launches were ~22 ms of cpu per
    step, the main thread's bottleneck on busy nodes). Resampling stays per image (shapes
    differ)."""
    device = samples[0]["values"].device
    num = len(samples)

    # per image scalar stats, on each image's own tensor (same kernels as the per image path)
    scaled, means, stds = [], [], []
    for sample in samples:
        values = sample["values"].float()
        values = values / values.max()
        scaled.append(values)
        means.append(values.mean())
        stds.append(values.std(correction=0))
    num_values = [v.numel() for v in scaled]
    num_runs = [sample["mask"].numel() for sample in samples]
    num_voxels = [math.prod(sample["shape"]) for sample in samples]
    total_values, total_runs, total_voxels = sum(num_values), sum(num_runs), sum(num_voxels)
    # the few per image counts go up in one copy, pinned so the host doesn't sync on it
    run_ends = list(itertools.accumulate(num_runs))
    meta = torch.tensor([run_ends, [0, *run_ends[:-1]]], dtype=torch.int64)
    if device.type == "cuda":
        meta = meta.pin_memory().to(device, non_blocking=True)
    run_ends, run_offsets = meta[0], meta[1]

    # z-score each image's segment in place with its stats: the same elementwise fp32 ops as
    # the per image path, and no extra pass over the values (a per element gather of the
    # stats costs ~1 GB of traffic on the side stream)
    values = torch.cat(scaled)
    stds = torch.stack(stds).clamp_min(1e-6)
    offset = 0
    for ii in range(num):
        segment = values[offset : offset + num_values[ii]]
        segment.sub_(means[ii]).div_(stds[ii])
        offset += num_values[ii]

    # run length decode of all masks at once (integer math, exact). each image's runs sum to
    # its voxel count, so the global cumsum of the concatenated runs is the flat index into the
    # concatenation of the dense volumes; the values are concatenated in the same order, so
    # the cumsum of the true run lengths is their flat value index
    lengths = torch.cat([sample["mask"] for sample in samples]).long()
    run_pos = torch.arange(total_runs, device=device)
    run_pos = run_pos - run_offsets[torch.bucketize(run_pos, run_ends, right=True)]
    true_lengths = lengths * (run_pos & 1)  # runs alternate false / true, false first
    voxel_starts = torch.cumsum(lengths, 0) - lengths
    value_starts = torch.cumsum(true_lengths, 0) - true_lengths
    index = torch.repeat_interleave(
        voxel_starts - value_starts, true_lengths, output_size=total_values
    )
    index += torch.arange(total_values, device=device)
    dense = values.new_zeros(total_voxels).scatter_(0, index, values)
    dense_mask = torch.zeros(total_voxels, dtype=torch.uint8, device=device)
    dense_mask.scatter_(0, index, 1)

    # resample, crop / pad into the zeroed batch and apply the mask
    images = torch.zeros(num, *GRID_SHAPE, device=device)
    masks = torch.zeros(num, *GRID_SHAPE, dtype=torch.uint8, device=device)
    offset = 0
    for ii, sample in enumerate(samples):
        shape = sample["shape"]
        spacing = sample["spacing"]
        image = dense[offset : offset + num_voxels[ii]].view(shape)
        mask = dense_mask[offset : offset + num_voxels[ii]].view(shape)
        offset += num_voxels[ii]
        resized_image = resample_image(image, spacing=spacing, target_spacing=TARGET_SPACING)
        mask = resample_image(mask, spacing=spacing, target_spacing=TARGET_SPACING, mode="nearest")
        src, dst = fit_slices(resized_image.shape, GRID_SHAPE)
        masks[ii][dst] = mask[src]
        if resized_image is image:
            images[ii][dst] = image[src]
        else:
            torch.mul(resized_image[src], mask[src], out=images[ii][dst])
    return {"image": images, "mask": masks, "path": [sample["path"] for sample in samples]}


def process_batch_per_image(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    # each sample is resampled to 1 mm and written straight into its slot of a preallocated
    # batch, cropped or zero padded to GRID_SHAPE. masks are uint8 (0/1)
    device = samples[0]["values"].device
    # only the padding around each sample's box gets zeroed (zero_outside)
    images = torch.empty(len(samples), *GRID_SHAPE, device=device)
    masks = torch.empty(len(samples), *GRID_SHAPE, dtype=torch.uint8, device=device)
    for ii, sample in enumerate(samples):
        spacing = sample["spacing"]

        # normalize values
        values = sample["values"].float()
        values = values / values.max()
        values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)

        # decode mask and make dense image
        image, mask = brle_scatter(sample["mask"], values, shape=sample["shape"])

        # resizing. the mask resamples as uint8, nearest keeps it 0/1
        resized_image = resample_image(image, spacing=spacing, target_spacing=TARGET_SPACING)
        mask = resample_image(mask, spacing=spacing, target_spacing=TARGET_SPACING, mode="nearest")

        # crop/pad into the batch and apply mask. without resizing the image is already zero
        # outside the mask
        src, dst = fit_slices(resized_image.shape, GRID_SHAPE)
        zero_outside(images[ii], dst)
        zero_outside(masks[ii], dst)
        masks[ii][dst] = mask[src]
        if resized_image is image:
            images[ii][dst] = image[src]
        else:
            torch.mul(resized_image[src], mask[src], out=images[ii][dst])
    return {"image": images, "mask": masks, "path": [sample["path"] for sample in samples]}


def brle_scatter(
    mask_rle: torch.Tensor, values: torch.Tensor, shape: tuple[int, int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    # dense mask and image (values in the mask's true voxels in order, zero elsewhere) from the
    # binary run length encoding, mask as uint8. same as brle_to_dense + masked_scatter_ into
    # zeros, but works on the runs and values instead of scanning the whole volume
    num_values = values.numel()
    lengths = mask_rle.long()
    starts = torch.cumsum(lengths, 0) - lengths
    true_starts, true_lengths = starts[1::2], lengths[1::2]
    # flat index of each value: its run's start plus its offset within the run
    run_shift = true_starts - (torch.cumsum(true_lengths, 0) - true_lengths)
    index = torch.repeat_interleave(run_shift, true_lengths, output_size=num_values)
    index += torch.arange(num_values, device=values.device)
    image = values.new_zeros(math.prod(shape)).scatter_(0, index, values)
    mask = torch.zeros(math.prod(shape), dtype=torch.uint8, device=values.device)
    mask.scatter_(0, index, 1)
    return image.reshape(shape), mask.reshape(shape)


def zero_outside(x: torch.Tensor, box: tuple[slice, ...]) -> None:
    # zero x everywhere outside box: the slabs before/after the box on each axis, within the box
    # on the previous axes
    for axis, sl in enumerate(box):
        lead = box[:axis]
        if sl.start > 0:
            x[(*lead, slice(0, sl.start))].zero_()
        if sl.stop < x.shape[axis]:
            x[(*lead, slice(sl.stop, None))].zero_()


def fit_slices(shape: tuple[int, ...], target_shape: tuple[int, ...]) -> tuple[tuple, tuple]:
    """Source and destination slices of fit_to_shape's pad (x/y centered, z top aligned)."""
    src, dst = [], []
    for axis, (size, target) in enumerate(zip(shape, target_shape)):
        diff = target - size
        before = diff // 2 if axis < 2 else diff
        n = min(size + before, target) - max(before, 0)
        src.append(slice(max(-before, 0), max(-before, 0) + n))
        dst.append(slice(max(before, 0), max(before, 0) + n))
    return tuple(src), tuple(dst)


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
