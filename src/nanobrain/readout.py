"""Eval embeddings read out from chosen encoder layers.

The frozen eval loads whatever class a checkpoint names in model_class, so a readout is a
ViTMAE3D subclass with no new weights. This module's main rewrites a trained ViTMAE3D
checkpoint to use it:

    python -m nanobrain.readout CKPT OUT --layers 3 6 9 12 --combine cat
    python -m nanobrain.readout CKPT OUT --layers 6 9 12 --global-layers 9 12
    python -m nanobrain.eval.run OUT OUT_DIR
"""

import argparse
from pathlib import Path

import torch
from jaxtyping import Float
from torch import Tensor

from .model import ViTMAE3D, patchify3d

MODEL_CLASS = "nanobrain.readout:ReadoutViTMAE3D"


class ReadoutViTMAE3D(ViTMAE3D):
    """ViTMAE3D whose eval embeddings come from chosen encoder blocks (1-indexed). Each block's
    output goes through the final norm, then the blocks are concatenated or averaged. Layers (12,)
    is exactly ViTMAE3D. global_layers, if set, are the blocks pooled for global_embed instead of
    layers. global_cls prepends the class token to the pooled global embedding."""

    @classmethod
    def from_config(cls, args: dict) -> "ReadoutViTMAE3D":
        model = super().from_config(args)
        readout = args["readout"]
        model.readout_layers = tuple(readout["layers"])
        # older readout checkpoints have no global_layers: global_embed pools the dense layers
        model.readout_global_layers = tuple(readout.get("global_layers") or readout["layers"])
        model.readout_combine = readout["combine"]
        model.readout_global_cls = readout["global_cls"]
        assert model.readout_combine in ("cat", "mean"), model.readout_combine
        layers = model.readout_layers + model.readout_global_layers
        assert all(1 <= layer <= len(model.blocks) for layer in layers)
        assert not model.readout_global_cls or model.class_tokens == 1
        return model

    def forward_layers(
        self,
        patches: Float[Tensor, "B N P"],
        coord: Float[Tensor, "B N 3"],
        layers: tuple[int, ...],
    ) -> Float[Tensor, "B R+N C"]:
        """forward_encoder, keeping the normed output of each of the given blocks."""
        B = patches.shape[0]
        x = self.patch_embed(patches)
        x = self.pos_embed(x, coord)
        if self.class_tokens:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        if self.hc_rate:
            x = x[:, :, None, :].expand(-1, -1, self.hc_rate, -1)
        outputs = []
        for layer, block in enumerate(self.blocks, start=1):
            x = block(x)
            if layer in layers:
                # with hyper-connections a layer's output is the sum of its streams, as at the end
                dtype = self.norm.weight.dtype
                outputs.append(self.norm(x.sum(dim=2, dtype=dtype) if self.hc_rate else x.to(dtype)))
            if layer == max(layers):
                break

        if self.readout_combine == "cat":
            return torch.cat(outputs, dim=-1)
        return torch.stack(outputs).mean(dim=0)

    def readout(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
        layers: tuple[int, ...],
    ) -> tuple[Float[Tensor, "B C"], Float[Tensor, "B Gx Gy Gz C"]]:
        """Class token and patch embeddings of the given blocks, one forward per image as in
        ViTMAE3D.dense_embed."""
        B = images.shape[0]
        assert tuple(images.shape[1:]) == tuple(self.grid_size)
        patch_grid_size = [size // self.patch_size for size in self.grid_size]
        num_outputs = len(layers) if self.readout_combine == "cat" else 1
        dim = num_outputs * self.embed_dim
        cls_embeds = images.new_zeros(B, dim)
        embeds = images.new_zeros(B, self.coord_grid.shape[1], dim)

        for ii in range(B):
            image_mask = mask[ii] > 0
            values = images[ii][image_mask]
            values = values / values.max()
            values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)
            image = torch.zeros_like(images[ii])
            image[image_mask] = values

            patches = patchify3d(image[None], self.patch_size)
            in_mask = patchify3d(image_mask[None], self.patch_size).any(dim=2)[0]
            x = self.forward_layers(patches[:, in_mask], self.coord_grid[:, in_mask], layers)
            if self.class_tokens:
                cls_embeds[ii] = x[0, 0].to(cls_embeds.dtype)
            embeds[ii, in_mask] = x[0, self.class_tokens :].to(embeds.dtype)

        embeds = embeds.reshape(B, *patch_grid_size, dim)
        return cls_embeds, embeds

    def dense_embed(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
    ) -> Float[Tensor, "B Gx Gy Gz C"]:
        return self.readout(images, mask, self.readout_layers)[1]

    def global_embed(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
    ) -> Float[Tensor, "B C"]:
        cls_embeds, embeds = self.readout(images, mask, self.readout_global_layers)
        embeds = embeds.flatten(1, 3)
        in_mask = patchify3d(mask > 0, self.patch_size).any(dim=2)
        embeds = (embeds * in_mask[..., None]).sum(dim=1) / in_mask.sum(dim=1, keepdim=True)
        if self.readout_global_cls:
            embeds = torch.cat([cls_embeds, embeds], dim=1)
        return embeds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    # blocks for global_embed, default the same as --layers
    parser.add_argument("--global-layers", type=int, nargs="+", default=None)
    parser.add_argument("--combine", choices=["cat", "mean"], default="cat")
    parser.add_argument("--global-cls", action="store_true")
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    assert checkpoint["model_class"] == "nanobrain.model:ViTMAE3D", checkpoint["model_class"]
    readout = {
        "layers": args.layers,
        "global_layers": args.global_layers,
        "combine": args.combine,
        "global_cls": args.global_cls,
    }
    # keep only what nanobrain.eval.run reads, no optimizer state
    checkpoint = {
        "model": checkpoint["model"],
        "args": {**checkpoint["args"], "readout": readout},
        "model_class": MODEL_CLASS,
        "git_sha": checkpoint["git_sha"],
        "epoch": checkpoint["epoch"],
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out_path)
