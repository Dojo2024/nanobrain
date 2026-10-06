import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from jaxtyping import Float, Int

from .modules import (
    Block,
    FracConnectionBlock,
    HyperConnectionBlock,
    LayerNorm,
    SeparablePosEmbed3D,
)


class ViTMAE3D(nn.Module):
    def __init__(
        self,
        grid_size: tuple[int, int, int] = (192, 240, 192),
        patch_size: int = 8,
        depth: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        decoder_depth: int = 4,
        decoder_embed_dim: int = 512,
        decoder_num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: int | float = 4,
        class_tokens: int = 1,
        drop_path_rate: float = 0.0,
        stats_head: bool = False,
        decoder_type: str = "self",
        hyper_connections: int = 0,
        hc_dynamic: bool = True,
        frac_connections: int = 0,
        fc_dynamic: bool = True,
    ):
        super().__init__()
        assert decoder_type in ("self", "cross"), decoder_type
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.class_tokens = class_tokens
        patch_grid_size = tuple(size // patch_size for size in grid_size)

        patch_dim = patch_size**3
        self.register_buffer("coord_grid", make_coord_grid(grid_size, patch_size), persistent=False)

        # encoder
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = SeparablePosEmbed3D(patch_grid_size, embed_dim)
        if class_tokens:
            self.cls_token = nn.Parameter(torch.empty(1, class_tokens, embed_dim))
        else:
            self.cls_token = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # hyper_connections: number of residual streams (Zhu et al., arXiv 2409.19606), 0 for the
        # plain residual. The encoder then carries [B, N, rate, D] and sums the streams at the end.
        # frac_connections: number of fractions the hidden state is split into (Zhu et al., arXiv
        # 2503.14125), 0 for the plain residual. The hidden state keeps its shape.
        assert not (hyper_connections and frac_connections), "hyper- or frac-connections, not both"
        self.hc_rate = hyper_connections
        if hyper_connections:
            block_cls = HyperConnectionBlock
            block_kwargs = lambda ii: dict(rate=hyper_connections, dynamic=hc_dynamic, layer_id=ii)
        elif frac_connections:
            block_cls = FracConnectionBlock
            block_kwargs = lambda ii: dict(rate=frac_connections, dynamic=fc_dynamic)
        else:
            block_cls = Block
            block_kwargs = lambda ii: {}
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[ii],
                    **block_kwargs(ii),
                )
                for ii in range(depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)

        # decoder
        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = SeparablePosEmbed3D(patch_grid_size, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_embed_dim))

        # self: mask tokens and visible embeddings attend jointly, as in MAE. cross: mask tokens
        # attend to the visible embeddings only, which stay fixed (CrossMAE), so the blocks run on
        # the mask tokens alone and each prediction is independent of the other mask tokens.
        self.decoder_type = decoder_type
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                )
                for ii in range(decoder_depth)
            ]
        )

        self.decoder_norm = LayerNorm(decoder_embed_dim)
        # stats_head: 2 extra outputs per patch, its in-mask intensity mean and log std
        self.stats_head = stats_head
        self.decoder_head = nn.Linear(decoder_embed_dim, patch_dim + (2 if stats_head else 0))
        self.init_weights()

    @classmethod
    def from_config(cls, args: dict) -> "ViTMAE3D":
        return cls(
            grid_size=tuple(args["grid_size"]),
            patch_size=args["patch_size"],
            **args["model_kwargs"],
        )

    def extra_repr(self):
        return f"{self.grid_size}, {self.patch_size}, class_tokens={self.class_tokens}"

    def init_weights(self):
        self.apply(_init_weights)
        if self.class_tokens:
            nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        # initialize the decoder head to zero to prevent hockey stick
        self.decoder_head.weight.data.mul_(0.0)

    def forward_encoder(
        self,
        patches: Float[Tensor, "B N P"],
        coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B R+N D"]:
        B, N, P = patches.shape
        assert coord.shape == (B, N, 3)

        x = self.patch_embed(patches)
        x = self.pos_embed(x, coord)

        if self.class_tokens:
            x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        if self.hc_rate:
            # the hyper hidden matrix starts as rate copies of each token
            x = x[:, :, None, :].expand(-1, -1, self.hc_rate, -1)
        for block in self.blocks:
            x = block(x)
        if self.hc_rate:
            x = x.sum(dim=2)
        x = self.norm(x)
        return x

    def forward_decoder(
        self,
        embeds: Float[Tensor, "B R+N D"],
        target_coord: Float[Tensor, "B M 3"],
    ) -> Float[Tensor, "B M P"]:
        B = embeds.shape[0]
        M = target_coord.shape[1]
        assert target_coord.shape == (B, M, 3)

        embeds = self.decoder_proj(embeds)

        mask_tokens = self.mask_token.expand(B, M, -1)
        mask_tokens = self.decoder_pos_embed(mask_tokens, target_coord)

        if self.decoder_type == "cross":
            x = mask_tokens
            for block in self.decoder_blocks:
                x = block(x, context=embeds)
        else:
            x = torch.cat([embeds, mask_tokens], dim=1)
            for block in self.decoder_blocks:
                x = block(x)
            x = x[:, -M:]

        x = self.decoder_norm(x)
        x = self.decoder_head(x)
        return x

    def forward(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
        num_visible: int,
        num_predict: int | None = None,
        num_samples: int = 1,
        with_state: bool = True,
        mask_groups: Int[Tensor, "K N"] | None = None,
        norm_target: bool = False,
        stats_weight: float = 1.0,
    ):
        B = images.shape[0]
        assert tuple(images.shape[1:]) == tuple(self.grid_size)

        patches = patchify3d(images, self.patch_size)
        mask_patches = patchify3d(mask, self.patch_size)
        coord = self.coord_grid.expand(B, -1, -1)

        # sample num_samples sequences per image, then flatten them into the batch
        # num_predict None predicts all remaining patches (dense decoding)
        seq_length = None if num_predict is None else num_visible + num_predict
        order = sample_sequences(mask_patches, num_samples, seq_length, mask_groups=mask_groups)
        num_predict = order.shape[2] - num_visible
        batch_ids = torch.arange(B, device=images.device)[:, None, None]
        patches = patches[batch_ids, order].flatten(0, 1)
        mask_patches = mask_patches[batch_ids, order].flatten(0, 1)
        coord = coord[batch_ids, order].flatten(0, 1)

        vis_patches, target_patches = torch.split(patches, [num_visible, num_predict], dim=1)
        vis_mask_patches, target_mask_patches = torch.split(
            mask_patches, [num_visible, num_predict], dim=1
        )
        vis_coord, target_coord = torch.split(coord, [num_visible, num_predict], dim=1)

        if norm_target:
            # MAE's norm_pix_loss: each target patch standardized by the mean and variance of its
            # in-mask voxels, so the loss is on local structure rather than intensity level
            count = target_mask_patches.sum(dim=-1, keepdim=True).clamp_min(1)
            mean = (target_mask_patches * target_patches).sum(dim=-1, keepdim=True) / count
            var = (target_mask_patches * (target_patches - mean) ** 2).sum(dim=-1, keepdim=True) / count
            target_stats = torch.cat([mean, 0.5 * torch.log(var + 1e-6)], dim=-1)
            target_patches = (target_patches - mean) / (var + 1e-6).sqrt()
        assert norm_target or not self.stats_head, "stats_head needs norm_target"

        embeds = self.forward_encoder(vis_patches, vis_coord)
        pred_patches = self.forward_decoder(embeds, target_coord)
        if self.stats_head:
            pred_patches, pred_stats = pred_patches.split([pred_patches.shape[-1] - 2, 2], dim=-1)
        loss = F.mse_loss(pred_patches, target_patches, reduction="none")
        loss = (target_mask_patches * loss).sum() / target_mask_patches.sum()
        if self.stats_head:
            # puts back the intensity level and spread that norm_target takes out of the target;
            # patches with under 2 in-mask voxels have no spread
            has_spread = (target_mask_patches.sum(dim=-1) > 1).to(loss.dtype)
            stats_loss = ((pred_stats - target_stats) ** 2).mean(dim=-1)
            loss = loss + stats_weight * (has_spread * stats_loss).sum() / has_spread.sum()

        if not with_state:
            return loss

        patch_ids = order.flatten(0, 1)
        image_ids = torch.arange(B, device=images.device).repeat_interleave(num_samples)
        vis_ids, target_ids = torch.split(patch_ids, [num_visible, num_predict], dim=1)

        state = {
            "vis_patches": vis_patches,
            "target_patches": target_patches,
            "vis_mask_patches": vis_mask_patches,
            "target_mask_patches": target_mask_patches,
            "embeds": embeds,
            "pred_patches": pred_patches,
            "vis_ids": vis_ids,
            "target_ids": target_ids,
            "image_ids": image_ids,
        }
        return loss, state

    def dense_embed(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
    ) -> Float[Tensor, "B Gx Gy Gz D"]:
        B = images.shape[0]
        assert tuple(images.shape[1:]) == tuple(self.grid_size)
        patch_grid_size = [size // self.patch_size for size in self.grid_size]
        embeds = images.new_zeros(B, self.coord_grid.shape[1], self.embed_dim)

        # one forward per image since the number of patches in the mask varies
        for ii in range(B):
            image_mask = mask[ii] > 0
            values = images[ii][image_mask]
            values = values / values.max()
            values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)
            image = torch.zeros_like(images[ii])
            image[image_mask] = values

            patches = patchify3d(image[None], self.patch_size)
            in_mask = patchify3d(image_mask[None], self.patch_size).any(dim=2)[0]
            x = self.forward_encoder(patches[:, in_mask], self.coord_grid[:, in_mask])
            embeds[ii, in_mask] = x[0, self.class_tokens :].to(embeds.dtype)

        embeds = embeds.reshape(B, *patch_grid_size, self.embed_dim)
        return embeds

    def global_embed(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
    ) -> Float[Tensor, "B D"]:
        embeds = self.dense_embed(images, mask).flatten(1, 3)
        in_mask = patchify3d(mask > 0, self.patch_size).any(dim=2)
        embeds = (embeds * in_mask[..., None]).sum(dim=1) / in_mask.sum(dim=1, keepdim=True)
        return embeds


def patchify3d(x: Tensor, patch_size: int = 8) -> Tensor:
    p = patch_size
    x = rearrange(x, "b (gx px) (gy py) (gz pz) -> b (gx gy gz) (px py pz)", px=p, py=p, pz=p)
    return x


def unpatchify3d(x: Tensor, grid_size: tuple[int, int, int], patch_size: int = 8) -> Tensor:
    p = patch_size
    gx, gy, gz = [size // patch_size for size in grid_size]
    x = rearrange(
        x,
        "b (gx gy gz) (px py pz) -> b (gx px) (gy py) (gz pz)",
        gx=gx,
        gy=gy,
        gz=gz,
        px=p,
        py=p,
        pz=p,
    )
    return x


def patches_to_volume(
    patches: Float[Tensor, "B L P"],
    patch_ids: Int[Tensor, "B L"],
    grid_size: tuple[int, int, int],
    patch_size: int = 8,
) -> Float[Tensor, "B X Y Z"]:
    B, L, P = patches.shape
    gx, gy, gz = [size // patch_size for size in grid_size]
    grid = patches.new_zeros(B, gx * gy * gz, P)
    batch_ids = torch.arange(B, device=patches.device)[:, None]
    grid[batch_ids, patch_ids] = patches
    images = unpatchify3d(grid, grid_size=grid_size, patch_size=patch_size)
    return images


def make_coord_grid(
    grid_size: tuple[int, int, int],
    patch_size: int = 8,
    device: torch.device | None = None,
) -> Tensor:
    patch_grid_size = tuple(size // patch_size for size in grid_size)
    grid_ids = [torch.arange(size, device=device) for size in patch_grid_size]
    coords = torch.stack(torch.meshgrid(*grid_ids, indexing="ij"), dim=-1)
    coords = coords.reshape(1, -1, 3)
    return coords


def make_mask_groups(
    patch_grid_size: tuple[int, int, int],
    group_shapes: list[list[int]],
) -> Int[Tensor, "K N"]:
    """Group id of every patch for each group shape [gx, gy, gz] in patches. Patches in one
    group are seen or hidden together: [1, 1, 30] is a full tube along the last axis, [4, 4, 4]
    a block of 4^3 patches."""
    grid = torch.stack(
        torch.meshgrid(*[torch.arange(size) for size in patch_grid_size], indexing="ij"), dim=-1
    ).reshape(-1, 3)
    group_ids = []
    for shape in group_shapes:
        shape = torch.tensor(list(shape))
        num_groups = [math.ceil(size / s) for size, s in zip(patch_grid_size, shape.tolist())]
        block = grid // shape
        group_ids.append((block[:, 0] * num_groups[1] + block[:, 1]) * num_groups[2] + block[:, 2])
    return torch.stack(group_ids)


def sample_sequences(
    mask_patches: torch.Tensor,
    num_samples: int,
    seq_length: int | None = None,
    min_mask_frac: float = 0.25,
    mask_groups: Int[Tensor, "K N"] | None = None,
) -> torch.Tensor:
    batch_size, num_patches, _ = mask_patches.shape
    mask_frac = mask_patches.mean(dim=2)
    # one independent shuffle per sequence, with patches outside the mask sorted last so
    # they are only used when a volume has too few mask patches
    scores = torch.rand(batch_size, num_samples, num_patches, device=mask_patches.device)
    if mask_groups is not None:
        # structured masks: each sequence picks one of the K groupings, then shuffles whole
        # groups, so a group's patches are contiguous in the order (seen or hidden together).
        # group ids are < num_patches, so a random permutation of num_patches ranks the groups
        choice = torch.randint(
            len(mask_groups), (batch_size, num_samples), device=mask_patches.device
        )
        group_ids = mask_groups[choice]
        group_rank = torch.rand(batch_size, num_samples, num_patches, device=mask_patches.device)
        group_rank = group_rank.argsort(dim=2).gather(2, group_ids)
        scores = (group_rank + scores) / num_patches
    scores = torch.where(mask_frac[:, None, :] >= min_mask_frac, scores, 2.0)
    if seq_length is None:
        # one host sync to get the longest mask sequence in the batch
        seq_length = (mask_frac >= min_mask_frac).sum(dim=1).max().item()
    order = scores.argsort(dim=2)[:, :, :seq_length]
    return order


# JAX ViT xavier uniform init
# https://github.com/facebookresearch/capi/blob/main/model.py
def _init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
        nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
