import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from brainmarks_smri.models import EmbeddingOutput, ImageInput, register_model
from einops import rearrange
from torch import Tensor
from jaxtyping import Float, Int

from .modules import Block, LayerNorm, SeparablePosEmbed3D
from .transforms import (
    crop_to_mask,
    fit_to_shape,
    normalize_values,
    reorient_to_ras,
    resample_to_spacing,
)

TARGET_SPACING = 1.0


class ViTMAE3D(nn.Module):
    name = "vit_mae_3d"

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
        norm_target: bool = False,
        grad_loss_weight: float = 0.0,
        fg_loss_weight: float = 0.0,
        dense_neighbor_pool: bool | str = False,
        decouple_readout: bool = False,
        brain_mask_pool: bool | float = False,
        **extra_kwargs,
    ):
        super().__init__()
        self.grid_size = grid_size
        # tuple for the brainmarks-smri model contract
        self.patch_size = (patch_size, patch_size, patch_size)
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
        self.decoder_sub = decoder_sub
        self.norm_target = norm_target
        self.grad_loss_weight = float(grad_loss_weight)
        self.fg_loss_weight = float(fg_loss_weight)
        self.dense_neighbor_pool = dense_neighbor_pool
        self.decouple_readout = bool(decouple_readout)
        self.brain_mask_pool = brain_mask_pool
        if decoder_sub > 1:
            assert patch_size % decoder_sub == 0, (patch_size, decoder_sub)
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
                    for ii in range(decoder_sub_depth)
                ]
            )
            self.decoder_sub_norm = LayerNorm(decoder_embed_dim)
            sub_coords = torch.stack(
                torch.meshgrid(*[torch.arange(decoder_sub)] * 3, indexing="ij"), dim=-1
            )
            self.register_buffer(
                "decoder_sub_coords", sub_coords.reshape(1, -1, 3), persistent=False
            )
            head_dim = (patch_size // decoder_sub) ** 3
            self.decoder_head = nn.Linear(decoder_embed_dim, head_dim)
        else:
            self.decoder_head = nn.Linear(decoder_embed_dim, patch_dim)
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
        if self.decoder_sub > 1:
            nn.init.xavier_uniform_(self.decoder_unshuffle.weight)
            if self.decoder_unshuffle.bias is not None:
                nn.init.constant_(self.decoder_unshuffle.bias, 0)
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

        readout_layers = getattr(self, "readout_layers", None)
        layer = getattr(self, "layer", None)
        if readout_layers is not None:
            outputs = []
            dtype = self.norm.weight.dtype
            for idx, block in enumerate(self.blocks, start=1):
                x = block(x)
                if idx in readout_layers:
                    outputs.append(self.norm(x.to(dtype)))
                if idx == max(readout_layers):
                    break
            combine = getattr(self, "readout_combine", "cat")
            if combine == "cat":
                return torch.cat(outputs, dim=-1)
            elif combine == "mean":
                return torch.stack(outputs).mean(dim=0)
            else:
                raise ValueError(f"Unknown readout combine: {combine}")
        elif layer is not None:
            for idx, block in enumerate(self.blocks, start=1):
                x = block(x)
                if idx == layer:
                    break
            x = self.norm(x)
            return x
        else:
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
        if self.decoder_sub > 1:
            s = self.decoder_sub
            x = self.decoder_norm(x)
            x = self.decoder_unshuffle(x).reshape(B, M * s**3, -1)
            sub_coord = target_coord[:, :, None, :] * s + self.decoder_sub_coords[None, :]
            x = self.decoder_sub_pos_embed(x, sub_coord.reshape(B, M * s**3, 3))
            for block in self.decoder_sub_blocks:
                x = block(x)
            x = self.decoder_head(self.decoder_sub_norm(x))
            return merge_subcubes(x.reshape(B, M, s**3, -1), self.patch_size[0], s)
        else:
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
    ):
        B = images.shape[0]
        assert tuple(images.shape[1:]) == tuple(self.grid_size)

        patches = patchify3d(images, self.patch_size[0])
        mask_patches = patchify3d(mask, self.patch_size[0])
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

        embeds = self.forward_encoder(vis_patches, vis_coord)
        pred_patches = self.forward_decoder(embeds, target_coord)
        if self.norm_target:
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_ref = (target_patches - mean) / (var + 1e-6) ** 0.5
        else:
            target_ref = target_patches
        loss = F.mse_loss(pred_patches, target_ref, reduction="none")
        if getattr(self, "fg_loss_weight", 0.0) > 0.0:
            fg_mask = (target_patches.abs() > 0.05).float()
            fg_weights = 1.0 + self.fg_loss_weight * fg_mask
            loss = (target_mask_patches * fg_weights * loss).sum() / (target_mask_patches * fg_weights).sum().clamp(min=1.0)
        else:
            loss = (target_mask_patches * loss).sum() / target_mask_patches.sum()

        if getattr(self, "grad_loss_weight", 0.0) > 0.0:
            p = self.patch_size[0]
            BM = pred_patches.shape[0] * pred_patches.shape[1]
            pred_vol = pred_patches.reshape(BM, p, p, p)
            tgt_vol = target_ref.reshape(BM, p, p, p)
            mask_vol = target_mask_patches.reshape(BM, p, p, p)
            dx_loss = F.mse_loss(
                pred_vol[:, 1:, :, :] - pred_vol[:, :-1, :, :],
                tgt_vol[:, 1:, :, :] - tgt_vol[:, :-1, :, :],
                reduction="none",
            )
            dy_loss = F.mse_loss(
                pred_vol[:, :, 1:, :] - pred_vol[:, :, :-1, :],
                tgt_vol[:, :, 1:, :] - tgt_vol[:, :, :-1, :],
                reduction="none",
            )
            dz_loss = F.mse_loss(
                pred_vol[:, :, :, 1:] - pred_vol[:, :, :, :-1],
                tgt_vol[:, :, :, 1:] - tgt_vol[:, :, :, :-1],
                reduction="none",
            )
            mx = mask_vol[:, 1:, :, :] * mask_vol[:, :-1, :, :]
            my = mask_vol[:, :, 1:, :] * mask_vol[:, :, :-1, :]
            mz = mask_vol[:, :, :, 1:] * mask_vol[:, :, :, :-1]
            grad_loss = (
                (mx * dx_loss).sum() / mx.sum().clamp(min=1.0)
                + (my * dy_loss).sum() / my.sum().clamp(min=1.0)
                + (mz * dz_loss).sum() / mz.sum().clamp(min=1.0)
            ) / 3.0
            loss = loss + self.grad_loss_weight * grad_loss

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

    def preprocess(self, sample: dict) -> dict:
        # native image -> model grid, for both pretraining and eval. expects the image cropped
        # to the mask, normalized and zero outside the mask
        image = sample["image"]
        mask = sample["mask"]
        affine = sample["affine"]
        image, mask, affine = reorient_to_ras(image, mask, affine)
        image, mask, affine = resample_to_spacing(image, mask, affine, TARGET_SPACING)
        image, mask, affine = fit_to_shape(image, mask, affine, self.grid_size)
        image = image * mask
        return {"image": image, "mask": mask, "affine": affine}

    def transform(self, sample: ImageInput) -> dict:
        # same crop and normalization as the pretraining data, see decode_sample in train.py
        image = torch.from_numpy(sample["image"].get_fdata(dtype=np.float32))
        mask = torch.from_numpy(np.asarray(sample["brain_mask"].dataobj) > 0)
        affine = sample["image"].affine
        image, mask, affine = crop_to_mask(image, mask, affine)
        normalized = torch.zeros_like(image)
        normalized[mask] = normalize_values(image[mask])
        return {"image": normalized, "mask": mask, "affine": affine}

    @torch.inference_mode()
    def forward_embeddings(
        self, samples: list[dict], return_dense: bool = False
    ) -> list[EmbeddingOutput]:
        device = self.coord_grid.device
        patch_grid_size = [size // self.patch_size[0] for size in self.grid_size]

        # one forward per image since the number of patches in the mask varies
        outputs = []
        for sample in samples:
            sample = {
                "image": sample["image"].to(device),
                "mask": sample["mask"].to(device),
                "affine": sample["affine"],
            }
            sample = self.preprocess(sample)
            patches = patchify3d(sample["image"][None], self.patch_size[0])
            in_mask = (patchify3d(sample["mask"][None], self.patch_size[0]) > 0).any(dim=2)[0]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                tokens = self.forward_encoder(patches[:, in_mask], self.coord_grid[:, in_mask])
            tokens = tokens[0, self.class_tokens :].float()

            mask_patches_float = patchify3d(sample["mask"][None].float(), self.patch_size[0])[0, in_mask]
            brain_ratio = mask_patches_float.mean(dim=-1)

            if getattr(self, "brain_mask_pool", False):
                pure_brain = brain_ratio >= 0.5
                if pure_brain.any():
                    target_tokens = tokens[pure_brain, -self.embed_dim :] if getattr(self, "decouple_readout", False) else tokens[pure_brain]
                    global_emb = target_tokens.mean(dim=0)
                else:
                    global_emb = tokens[:, -self.embed_dim :].mean(dim=0)
            elif getattr(self, "dual_global_readout", False) and getattr(self, "readout_layers", None) is not None and len(self.readout_layers) >= 2:
                global_emb = tokens[:, -2 * self.embed_dim :].mean(dim=0)
            elif getattr(self, "decouple_readout", False) and getattr(self, "readout_layers", None) is not None:
                global_emb = tokens[:, -self.embed_dim :].mean(dim=0)
            else:
                global_emb = tokens.mean(dim=0)

            output = EmbeddingOutput(
                global_embedding=global_emb,
                dense_embedding=None,
                dense_affine=None,
                dense_mask=None,
            )
            if return_dense:
                dim = tokens.shape[-1]
                dense = tokens.new_zeros(len(in_mask), dim)
                dense[in_mask] = tokens
                dense_grid = dense.reshape(*patch_grid_size, dim)
                if dense_grid.shape[-1] > 4096 and dense_grid.shape[-1] % 2 == 0:
                    dense_grid = dense_grid.reshape(*patch_grid_size, -1, 2).mean(dim=-1)
                if getattr(self, "dense_neighbor_pool", False):
                    mode = str(self.dense_neighbor_pool).lower()
                    sem_grid = dense_grid[..., -self.embed_dim :]
                    feat_5d = sem_grid.permute(3, 0, 1, 2).unsqueeze(0)
                    mask_5d = in_mask.reshape(1, 1, *patch_grid_size).float()
                    if mode in ["pyramid", "spp"]:
                        s3 = F.avg_pool3d(feat_5d, kernel_size=3, stride=1, padding=1) * 27.0
                        c3 = F.avg_pool3d(mask_5d, kernel_size=3, stride=1, padding=1) * 27.0
                        m3 = (s3 / c3.clamp(min=1.0)).squeeze(0).permute(1, 2, 3, 0)
                        s5 = F.avg_pool3d(feat_5d, kernel_size=5, stride=1, padding=2) * 125.0
                        c5 = F.avg_pool3d(mask_5d, kernel_size=5, stride=1, padding=2) * 125.0
                        m5 = (s5 / c5.clamp(min=1.0)).squeeze(0).permute(1, 2, 3, 0)
                        m3 = m3.reshape(*patch_grid_size, -1, 2).mean(dim=-1)
                        m5 = m5.reshape(*patch_grid_size, -1, 2).mean(dim=-1)
                        nbr_mean = torch.cat([m3, m5], dim=-1)
                    else:
                        nbr_sum = F.avg_pool3d(feat_5d, kernel_size=3, stride=1, padding=1) * 27.0
                        nbr_cnt = F.avg_pool3d(mask_5d, kernel_size=3, stride=1, padding=1) * 27.0
                        nbr_mean = (nbr_sum / nbr_cnt.clamp(min=1.0)).squeeze(0).permute(1, 2, 3, 0)
                        if dense_grid.shape[-1] + nbr_mean.shape[-1] > 3840:
                            nbr_mean = nbr_mean.reshape(*patch_grid_size, -1, 2).mean(dim=-1)
                    if dense_grid.shape[-1] + nbr_mean.shape[-1] > 4096:
                        nbr_mean = nbr_mean.reshape(*patch_grid_size, -1, 2).mean(dim=-1)
                    nbr_mean = nbr_mean * in_mask.reshape(*patch_grid_size, 1)
                    dense_grid = torch.cat([dense_grid, nbr_mean], dim=-1)
                output["dense_embedding"] = dense_grid
                output["dense_affine"] = sample["affine"]
                output["dense_mask"] = in_mask.reshape(patch_grid_size)
            outputs.append(output)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return outputs


@register_model
def vit_mae_3d(
    checkpoint: str,
    layer: int | None = None,
    readout_layers: list[int] | tuple[int, ...] | str | None = None,
    readout_combine: str = "cat",
    dense_neighbor_pool: bool | str | None = None,
    decouple_readout: bool | None = None,
    brain_mask_pool: bool | None = None,
    dual_global_readout: bool | None = None,
) -> ViTMAE3D:
    # brainmarks-smri builder, run with `--overrides 'imports=[nanobrain.model]'`
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = ViTMAE3D.from_config(ckpt["args"])
    model.load_state_dict(ckpt["model"])
    # Benchmark Rule for nanobrain: evaluate strictly on the last layer (no intermediate layers)
    readout_layers = None
    layer = None
    model.readout_layers = None
    model.layer = None
    if dense_neighbor_pool is not None:
        model.dense_neighbor_pool = dense_neighbor_pool
    if decouple_readout is not None:
        model.decouple_readout = bool(decouple_readout)
    if brain_mask_pool is not None:
        model.brain_mask_pool = bool(brain_mask_pool)
    if dual_global_readout is not None:
        model.dual_global_readout = bool(dual_global_readout)
    return model


def merge_subcubes(x: Tensor, patch_size: int, sub: int) -> Tensor:
    q = patch_size // sub
    return rearrange(
        x,
        "b m (ox oy oz) (qx qy qz) -> b m (ox qx oy qy oz qz)",
        ox=sub, oy=sub, oz=sub, qx=q, qy=q, qz=q,
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
