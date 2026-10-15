import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from jaxtyping import Float, Int

from .modules import Block, LayerNorm, SeparablePosEmbed3D


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
        decoder_sub: int = 1,
        decoder_sub_depth: int = 1,
    ):
        super().__init__()
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

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[ii],
                )
                for ii in range(depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)

        # decoder
        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = SeparablePosEmbed3D(patch_grid_size, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, decoder_embed_dim))

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

        # decoder_sub (pixel-unshuffle sub-token decoder): each predicted coarse patch
        # is split into decoder_sub^3 sub-cubes, each decoded as its own token on the finer grid.
        assert patch_size % decoder_sub == 0, (patch_size, decoder_sub)
        self.decoder_sub = decoder_sub
        head_dim = patch_dim
        if decoder_sub > 1:
            self.decoder_unshuffle = nn.Linear(decoder_embed_dim, decoder_sub**3 * decoder_embed_dim)
            sub_grid_size = tuple(size * decoder_sub for size in patch_grid_size)
            self.decoder_sub_pos_embed = SeparablePosEmbed3D(sub_grid_size, decoder_embed_dim)
            self.decoder_sub_blocks = nn.ModuleList(
                [
                    Block(
                        dim=decoder_embed_dim,
                        num_heads=decoder_num_heads,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        mlp_ratio=mlp_ratio,
                    )
                    for _ in range(decoder_sub_depth)
                ]
            )
            self.decoder_sub_norm = LayerNorm(decoder_embed_dim)
            offsets = torch.stack(
                torch.meshgrid(*[torch.arange(decoder_sub)] * 3, indexing="ij"), dim=-1
            ).reshape(-1, 3)
            self.register_buffer("sub_offsets", offsets, persistent=False)
            head_dim = (patch_size // decoder_sub) ** 3

        self.decoder_head = nn.Linear(decoder_embed_dim, head_dim)
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

        for block in self.blocks:
            x = block(x)
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

        x = torch.cat([embeds, mask_tokens], dim=1)

        for block in self.decoder_blocks:
            x = block(x)

        x = x[:, -M:]
        x = self.decoder_norm(x)
        if self.decoder_sub > 1:
            s = self.decoder_sub
            x = self.decoder_unshuffle(x).reshape(B, M * s**3, -1)
            sub_coord = target_coord[:, :, None, :] * s + self.sub_offsets
            x = self.decoder_sub_pos_embed(x, sub_coord.reshape(B, M * s**3, 3))
            for block in self.decoder_sub_blocks:
                x = block(x)
            x = self.decoder_head(self.decoder_sub_norm(x))
            return merge_subcubes(x.reshape(B, M, s**3, -1), self.patch_size, s)
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
        norm_target: bool = False,
    ):
        B = images.shape[0]
        assert tuple(images.shape[1:]) == tuple(self.grid_size)

        patches = patchify3d(images, self.patch_size)
        mask_patches = patchify3d(mask, self.patch_size)
        coord = self.coord_grid.expand(B, -1, -1)

        # sample num_samples sequences per image, then flatten them into the batch
        # num_predict None predicts all remaining patches (dense decoding)
        seq_length = None if num_predict is None else num_visible + num_predict
        order = sample_sequences(mask_patches, num_samples, seq_length)
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
            mask_val = target_mask_patches.to(target_patches.dtype)
            count = mask_val.sum(dim=-1, keepdim=True)
            mean = (mask_val * target_patches).sum(dim=-1, keepdim=True) / count.clamp_min(1)
            var = (mask_val * (target_patches - mean) ** 2).sum(dim=-1, keepdim=True) / count.clamp_min(1)
            std = torch.sqrt(var + 1e-6)
            target_patches = torch.where(mask_val > 0, (target_patches - mean) / std, 0.0)

        embeds = self.forward_encoder(vis_patches, vis_coord)
        pred_patches = self.forward_decoder(embeds, target_coord)
        loss = F.mse_loss(pred_patches, target_patches, reduction="none")
        loss = (target_mask_patches * loss).sum() / target_mask_patches.sum()

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
    ) -> Float[Tensor, "B Ex Ey Ez D"]:
        """Patch embeddings on the eval's 8 mm grid, which the frozen eval asserts. Other patch
        sizes are interpolated onto it (to_eval_grid); 8 mm is dense_embed_native unchanged."""
        embeds = self.dense_embed_native(images, mask)
        return to_eval_grid(embeds, patchify3d(mask > 0, self.patch_size).any(dim=2), self.patch_size)

    def dense_embed_native(
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
        embeds = self.dense_embed_native(images, mask).flatten(1, 3)
        in_mask = patchify3d(mask > 0, self.patch_size).any(dim=2)
        embeds = (embeds * in_mask[..., None]).sum(dim=1) / in_mask.sum(dim=1, keepdim=True)
        return embeds


# the frozen eval probes an 8 mm patch grid (eval/probe.py PATCH_GRID_SHAPE, 24x30x24)
EVAL_PATCH_SIZE = 8


def to_eval_grid(
    embeds: Float[Tensor, "B Gx Gy Gz D"],
    in_mask: torch.Tensor,
    patch_size: int,
) -> Float[Tensor, "B Ex Ey Ez D"]:
    """Trilinear interpolation of a patch_size grid onto the 8 mm eval grid, normalized by the
    interpolated in-mask weight so empty patches (zeros) do not dim the edge of the head.
    align_corners=False maps patch centres onto patch centres for any patch size."""
    if patch_size == EVAL_PATCH_SIZE:
        return embeds
    B, *grid, D = embeds.shape
    eval_grid = [size * patch_size // EVAL_PATCH_SIZE for size in grid]
    weight = in_mask.reshape(B, 1, *grid).to(embeds.dtype)
    x = torch.cat([embeds.permute(0, 4, 1, 2, 3) * weight, weight], dim=1)
    x = F.interpolate(x, size=eval_grid, mode="trilinear", align_corners=False)
    x, weight = x[:, :D], x[:, D:]
    x = torch.where(weight > 0, x / weight.clamp_min(1e-6), 0.0)
    return x.permute(0, 2, 3, 4, 1)


def split_subcubes(x: Float[Tensor, "B M P"], patch_size: int, sub: int) -> Float[Tensor, "B M S Q"]:
    """Voxels of each patch (patchify3d order) regrouped into its sub^3 sub-cubes, in
    meshgrid(ij) order of the sub-cube offsets, each in patchify3d order."""
    q = patch_size // sub
    return rearrange(
        x,
        "b m (ox qx oy qy oz qz) -> b m (ox oy oz) (qx qy qz)",
        ox=sub,
        oy=sub,
        oz=sub,
        qx=q,
        qy=q,
        qz=q,
    )


def merge_subcubes(x: Float[Tensor, "B M S Q"], patch_size: int, sub: int) -> Float[Tensor, "B M P"]:
    """Inverse of split_subcubes."""
    q = patch_size // sub
    return rearrange(
        x,
        "b m (ox oy oz) (qx qy qz) -> b m (ox qx oy qy oz qz)",
        ox=sub,
        oy=sub,
        oz=sub,
        qx=q,
        qy=q,
        qz=q,
    )


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


def sample_sequences(
    mask_patches: torch.Tensor,
    num_samples: int,
    seq_length: int | None = None,
    min_mask_frac: float = 0.25,
) -> torch.Tensor:
    batch_size, num_patches, _ = mask_patches.shape
    mask_frac = mask_patches.mean(dim=2)
    # one independent shuffle per sequence, with patches outside the mask sorted last so
    # they are only used when a volume has too few mask patches
    scores = torch.rand(batch_size, num_samples, num_patches, device=mask_patches.device)
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
