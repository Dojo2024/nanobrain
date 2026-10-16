import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from jaxtyping import Bool, Float, Int

from .modules import (
    Block,
    FracConnectionBlock,
    HyperConnectionBlock,
    LayerNorm,
    ManifoldHyperConnectionBlock,
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
        hc_bf16: bool = False,
        mhc: int = 0,
        decoder_sub: int = 1,
        decoder_sub_depth: int = 1,
        octant_head: bool = False,
        octant_sub_depth: int = 1,
        layout_head: bool = False,
        global_loss: str = "none",
        global_hidden_dim: int = 2048,
        global_proj_dim: int = 256,
        sigreg_slices: int = 1024,
        objective: str = "mae",
        siam_stoch: int = 32,
        siam_discrete: int = 32,
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
        # hc_bf16: store their hidden state in bf16 between branches, as the frac-connections
        # paper's code does; the mixing is computed in fp32 and the encoder output is fp32 again.
        # mhc: number of streams for manifold-constrained hyper-connections (Xie et al., arXiv
        # 2512.24880), 0 off. Same [B, N, rate, D] hidden matrix and stream sum as hyper_connections.
        assert sum(map(bool, (hyper_connections, frac_connections, mhc))) <= 1, "one connection type"
        assert not hc_bf16 or hyper_connections or frac_connections or mhc, "hc_bf16 needs hyper/frac/mhc"
        stream_dtype = torch.bfloat16 if hc_bf16 else None
        self.hc_rate = hyper_connections or mhc
        if mhc:
            block_cls = ManifoldHyperConnectionBlock
            block_kwargs = lambda ii: dict(rate=mhc, layer_id=ii, stream_dtype=stream_dtype)
        elif hyper_connections:
            block_cls = HyperConnectionBlock
            block_kwargs = lambda ii: dict(
                rate=hyper_connections, dynamic=hc_dynamic, layer_id=ii, stream_dtype=stream_dtype
            )
        elif frac_connections:
            block_cls = FracConnectionBlock
            block_kwargs = lambda ii: dict(
                rate=frac_connections, dynamic=fc_dynamic, stream_dtype=stream_dtype
            )
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

        # decoder_sub (P14 (ii), pixel-unshuffle decoder, after Scale-MAE / LocalMIM): each predicted
        # patch is split into decoder_sub^3 sub-cubes, each decoded as its own token on the finer
        # grid: Linear(Dd -> sub^3 Dd), the sub-cube's position, decoder_sub_depth blocks over all
        # sub-tokens, then a head over the sub-cube's voxels. The target is the same voxels
        # (norm_target still standardizes the whole parent patch). 1 is the plain decoder head.
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
                    for ii in range(decoder_sub_depth)
                ]
            )
            self.decoder_sub_norm = LayerNorm(decoder_embed_dim)
            offsets = torch.stack(
                torch.meshgrid(*[torch.arange(decoder_sub)] * 3, indexing="ij"), dim=-1
            ).reshape(-1, 3)
            self.register_buffer("sub_offsets", offsets, persistent=False)
            head_dim = (patch_size // decoder_sub) ** 3

        # P2: octant_head — eval-only dense readout that uses the decoder_sub pipeline to emit
        # sub^3 distinct vectors per encoder token, giving native 8 mm resolution from a coarse
        # encoder. Shares decoder_sub's pos embed and blocks (trained during MAE training).
        self.octant_head = octant_head
        if octant_head:
            assert decoder_sub > 1, "octant_head needs decoder_sub > 1"

        # stats_head: 2 extra outputs per patch, its in-mask intensity mean and log std
        # layout_head (P14 (i)): 16 extra outputs per patch, the raw in-mask mean and log std of each
        # of its 8 octant sub-cubes, i.e. where intensity sits inside the patch
        assert not (stats_head and layout_head), "one auxiliary head"
        assert decoder_sub == 1 or not (stats_head or layout_head), "stats/layout heads need decoder_sub 1"
        assert not octant_head or decoder_sub > 1, "octant_head needs decoder_sub > 1"
        self.stats_head = stats_head
        self.layout_head = layout_head
        num_aux = 2 if stats_head else 16 if layout_head else 0
        # SiamJEPA (Yamada, arXiv 2607.04044, github.com/oist/SiamJEPA): two students with disjoint
        # visible patches predict an EMA teacher's features of patches hidden from both, through
        # the decoder with a feature head; a latent Z couples the two students (forward_siamjepa)
        assert objective in ("mae", "siamjepa"), objective
        self.objective = objective
        if objective == "siamjepa":
            assert class_tokens == 1 and decoder_sub == 1 and decoder_type == "self", "siamjepa"
            assert not (stats_head or layout_head or self.hc_rate or frac_connections), "siamjepa"
            assert global_loss == "none", "siamjepa"
            head_dim = embed_dim
            E, Z = embed_dim, siam_stoch * siam_discrete
            self.siam_stoch, self.siam_discrete = siam_stoch, siam_discrete
            # the paper's linear map h (ca3 in the code), on every student token
            self.siam_h = nn.Linear(E, E)
            self.siam_projector = nn.Sequential(
                nn.Linear(E, E), nn.BatchNorm1d(E), nn.ReLU(),
                nn.Linear(E, E), nn.BatchNorm1d(E), nn.ReLU(),
                nn.Linear(E, E),
            )
            self.siam_posterior = nn.Sequential(
                nn.Linear(2 * E, 2 * E), nn.ReLU(), nn.Linear(2 * E, Z)
            )
            self.siam_prior = nn.Sequential(nn.Linear(E, 2 * E), nn.ReLU(), nn.Linear(2 * E, Z))
            self.siam_z_embed = nn.Linear(Z, decoder_embed_dim)
        self.decoder_head = nn.Linear(decoder_embed_dim, head_dim + num_aux)

        # global_loss (L8): a loss on the mean of each sequence's visible encoder tokens (the same
        # patch mean global_embed reads), through a projector that the eval never uses. The
        # num_samples sequences of one image are its views.
        #   sigreg: LeJEPA (Balestriero & LeCun, arXiv 2511.08544): pull an image's views to their
        #     mean, and push each view's batch of projections toward an isotropic Gaussian (SIGReg)
        #   infonce: COJEPA's contrastive term (Mager & Hansen, arXiv 2607.11962), between views 1 and 2
        assert global_loss in ("none", "sigreg", "infonce"), global_loss
        self.global_loss = global_loss
        self.sigreg_slices = sigreg_slices
        if global_loss != "none":
            self.global_proj = nn.Sequential(
                nn.Linear(embed_dim, global_hidden_dim),
                LayerNorm(global_hidden_dim),
                nn.GELU(),
                nn.Linear(global_hidden_dim, global_hidden_dim),
                LayerNorm(global_hidden_dim),
                nn.GELU(),
                nn.Linear(global_hidden_dim, global_proj_dim),
            )

        # A2: Fourier spectral loss (SL-MAE, Ergün et al., arXiv 2606.13315), see spectral_loss.
        # High-pass radial mask for per-patch 3D rfft. Shape (Px, Py, Pz//2+1).
        # Weight = 1 - exp(-r² / (2 * σ²)) with r in cycles per patch and σ = 1: DC gets 0, r = 1
        # gets .39, r = 2 gets .86, r >= 3 about 1. At patch 16 the same σ is a lower cutoff in mm.
        freq_shape = (patch_size, patch_size, patch_size // 2 + 1)
        freq_coords = torch.stack(
            torch.meshgrid(
                *[torch.arange(s) for s in freq_shape],
                indexing="ij",
            ),
            dim=-1,
        ).float()
        # Wrap frequencies: indices > half map to negative frequencies.
        for ii, s in enumerate(freq_shape[:2]):  # first two dims are full FFT
            freq_coords[..., ii] = torch.where(
                freq_coords[..., ii] > s // 2,
                freq_coords[..., ii] - s,
                freq_coords[..., ii],
            )
        # Last dim (rfft) is always non-negative, no wrapping.
        freq_radius = freq_coords.norm(dim=-1)
        highpass_mask = 1.0 - torch.exp(-0.5 * freq_radius ** 2)
        self.register_buffer("spectral_mask", highpass_mask, persistent=False)

        self.init_weights()
        if objective == "siamjepa":
            # EMA teacher: a frozen copy of the encoder, moved toward it by update_teacher
            parts = {name: copy.deepcopy(getattr(self, name)) for name in TEACHER_PARTS}
            self.teacher = nn.ModuleDict(parts)
            self.teacher_cls_token = nn.Parameter(self.cls_token.detach().clone())
            self.teacher.requires_grad_(False)
            self.teacher_cls_token.requires_grad_(False)

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
        # initialize the decoder head to zero to prevent hockey stick. Not for SiamJEPA: its
        # predictions are L2-normalized, and normalizing zeros gives ~1e12 gradients at step 0
        if self.objective == "mae":
            self.decoder_head.weight.data.mul_(0.0)

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """SiamJEPA: teacher = momentum * teacher + (1 - momentum) * student, after each step."""
        student = [p for name in TEACHER_PARTS for p in getattr(self, name).parameters()]
        teacher = list(self.teacher.parameters())
        assert len(student) == len(teacher)
        student, teacher = student + [self.cls_token], teacher + [self.teacher_cls_token]
        torch._foreach_mul_(teacher, momentum)
        torch._foreach_add_(teacher, student, alpha=1 - momentum)

    @torch.no_grad()
    def forward_teacher(
        self,
        patches: Float[Tensor, "B N P"],
        coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B N D"]:
        teacher = self.teacher
        x = teacher["pos_embed"](teacher["patch_embed"](patches), coord)
        x = torch.cat([self.teacher_cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        for block in teacher["blocks"]:
            x = block(x)
        return teacher["norm"](x)[:, 1:]

    def forward_siamjepa(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
        num_visible: int,
        num_predict: int,
        kl_scale: float,
        shuffle_teacher: bool,
    ) -> Float[Tensor, ""]:
        """SiamJEPA loss, following github.com/oist/SiamJEPA (models_siamjepa.py):
        - one shuffle of the in-mask patches per image: student view 1 sees the first num_visible,
          view 2 the next num_visible, and the targets are the next num_predict, hidden from both
          (random_masking_dual without its square block). One encoder runs both views.
        - the EMA teacher sees all of those patches; its final-norm patch features are the targets.
          With shuffle_teacher (Random Shuffle Teacher) they are permuted per image and never put
          back, so each target is another patch's feature.
        - predictor: student tokens through h, decoder_proj, a token embedding Z, mask tokens at
          the target positions, the decoder blocks, decoder_head to the teacher's width.
          Loss 2 - 2 cos(prediction, target) over in-mask targets, both views.
        - Z: 32 categorical variables of 32 classes, straight-through sample of the posterior
          q(Z | proj(h(cls own view)), proj(cls other view)); prior p(Z | sg(proj(h(cls own)))).
          KL balanced 0.2 posterior / 0.8 prior, free bits 0.1 per view, weight kl_scale."""
        B = images.shape[0]
        V, P = num_visible, num_predict
        patches = patchify3d(images, self.patch_size)
        mask_patches = patchify3d(mask, self.patch_size)
        coord = self.coord_grid.expand(B, -1, -1)
        order = sample_sequences(mask_patches, 1, 2 * V + P)[:, 0]
        batch_ids = torch.arange(B, device=images.device)[:, None]
        patches, coord = patches[batch_ids, order], coord[batch_ids, order]
        # the same in-mask rule as sample_sequences: volumes with too few mask patches fill the
        # tail of the order with outside patches, which are not targets
        valid = mask_patches[batch_ids, order[:, 2 * V :]].mean(dim=-1) >= 0.25

        target = self.forward_teacher(patches, coord)
        if shuffle_teacher:
            perm = torch.rand(B, target.shape[1], device=images.device).argsort(dim=1)
            target = target[batch_ids, perm]
        target = target[:, 2 * V :]

        views = torch.cat([patches[:, :V], patches[:, V : 2 * V]])
        embeds = self.forward_encoder(views, torch.cat([coord[:, :V], coord[:, V : 2 * V]]))
        hidden = self.siam_h(embeds)

        # Z for view 1 sees view 2's class token and vice versa (rows B: are view 2)
        own = self.siam_projector(hidden[:, 0])
        other = self.siam_projector(embeds[:, 0].roll(B, dims=0))
        shape = (2 * B, self.siam_stoch, self.siam_discrete)
        post = self.siam_posterior(torch.cat([own, other], dim=-1))
        post = post.float().clamp(-20, 20).view(shape)
        prior = self.siam_prior(own.detach()).float().clamp(-20, 20).view(shape)
        post, prior = post.log_softmax(dim=-1), prior.log_softmax(dim=-1)
        gumbel = -torch.log(-torch.log(torch.rand_like(post).clamp_min(1e-20)))
        one_hot = F.one_hot((post + gumbel).argmax(dim=-1), self.siam_discrete).to(post.dtype)
        z = one_hot + post.exp() - post.exp().detach()
        kl_post = (post.exp() * (post - prior.detach())).sum(dim=(1, 2))
        kl_prior = (post.exp().detach() * (post.detach() - prior)).sum(dim=(1, 2))
        kl = 0.2 * kl_post + 0.8 * kl_prior
        kl_loss = (kl[:B].mean().clamp_min(0.1) + kl[B:].mean().clamp_min(0.1)) / 2

        x = self.decoder_proj(hidden)
        z_token = self.siam_z_embed(z.flatten(1).to(x.dtype))[:, None]
        target_coord = coord[:, 2 * V :].repeat(2, 1, 1)
        mask_tokens = self.decoder_pos_embed(self.mask_token.expand(2 * B, P, -1), target_coord)
        x = torch.cat([z_token, x, mask_tokens], dim=1)
        for block in self.decoder_blocks:
            x = block(x)
        pred = self.decoder_head(self.decoder_norm(x[:, -P:]))

        pred = F.normalize(pred.float(), dim=-1)
        target = F.normalize(target.float(), dim=-1).repeat(2, 1, 1)
        sim = 2 - 2 * (pred * target).sum(dim=-1).clamp(-1, 1)
        weight = valid.repeat(2, 1).to(sim.dtype)
        loss = (sim * weight).sum() / weight.sum().clamp_min(1)
        return loss + kl_scale * kl_loss

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
        dtype = self.norm.weight.dtype
        if self.hc_rate:
            x = x.sum(dim=2, dtype=dtype)
        x = self.norm(x.to(dtype))
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
        mask_groups: Int[Tensor, "K N"] | None = None,
        norm_target: bool = False,
        stats_weight: float = 1.0,
        layout_weight: float = 0.1,
        global_weight: float = 0.1,
        sigreg_lambda: float = 0.05,
        infonce_tau: float = 0.07,
        spectral_weight: float = 0.0,
        siam_kl_scale: float = 0.01,
        shuffle_teacher: bool = False,
    ):
        if self.objective == "siamjepa":
            loss = self.forward_siamjepa(
                images, mask, num_visible, num_predict, siam_kl_scale, shuffle_teacher
            )
            return loss if not with_state else (loss, {})
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

        if self.layout_head:
            # raw in-mask mean and log std of each octant sub-cube, before norm_target removes the
            # patch's level; octants with under 2 in-mask voxels have no spread and are skipped
            octant_values = split_subcubes(target_patches, self.patch_size, 2)
            octant_mask = split_subcubes(target_mask_patches, self.patch_size, 2)
            count = octant_mask.sum(dim=-1)
            mean = (octant_mask * octant_values).sum(dim=-1) / count.clamp_min(1)
            var = (octant_mask * (octant_values - mean[..., None]) ** 2).sum(dim=-1) / count.clamp_min(1)
            layout_target = torch.cat([mean, 0.5 * torch.log(var + 1e-6)], dim=-1)
            layout_valid = (count > 1).to(mean.dtype).repeat(1, 1, 2)

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
        if self.layout_head:
            pred_patches, pred_layout = pred_patches.split([pred_patches.shape[-1] - 16, 16], dim=-1)
        loss = F.mse_loss(pred_patches, target_patches, reduction="none")
        loss = (target_mask_patches * loss).sum() / target_mask_patches.sum()
        # A2: per-patch spectral loss, patches weighted by their in-mask voxel count
        if spectral_weight > 0:
            spec_loss = spectral_loss(
                pred_patches, target_patches, target_mask_patches, self.spectral_mask, self.patch_size
            )
            patch_voxels = target_mask_patches.sum(dim=-1).to(spec_loss.dtype)
            loss = loss + spectral_weight * (spec_loss * patch_voxels).sum() / patch_voxels.sum()
        if self.layout_head:
            layout_loss = layout_valid * (pred_layout - layout_target) ** 2
            loss = loss + layout_weight * layout_loss.sum() / layout_valid.sum().clamp_min(1)
        # training only: train.py's end-of-epoch plot calls the model in eval mode with 1 sequence
        if self.global_loss != "none" and self.training:
            assert num_samples >= 2, "the global loss compares the sequences of one image"
            pooled = embeds[:, self.class_tokens :].mean(dim=1)
            z = self.global_proj(pooled).float().reshape(B, num_samples, -1)
            if self.global_loss == "sigreg":
                # LeJEPA: (1 - lambda) * invariance to the views' mean + lambda * SIGReg per view
                inv = (z - z.mean(dim=1, keepdim=True)).square().mean()
                sig = torch.stack([sigreg(z[:, ii], self.sigreg_slices) for ii in range(num_samples)])
                global_term = (1 - sigreg_lambda) * inv + sigreg_lambda * sig.mean()
            else:
                z1, z2 = F.normalize(z[:, 0], dim=-1), F.normalize(z[:, 1], dim=-1)
                logits = z1 @ z2.T / infonce_tau
                labels = torch.arange(B, device=z.device)
                global_term = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            loss = loss + global_weight * global_term
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
    ) -> Float[Tensor, "B Ex Ey Ez D"]:
        """Patch embeddings on the eval's 8 mm grid, which the frozen eval asserts. Other patch
        sizes are interpolated onto it (to_eval_grid); 8 mm is dense_embed_native unchanged.
        When octant_head is enabled, uses decoder_sub pipeline for native 8 mm resolution."""
        if self.octant_head:
            return self.dense_embed_native_octants(images, mask)
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

    def dense_embed_native_octants(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
    ) -> Float[Tensor, "B Fx Fy Fz D"]:
        """P2: dense embeddings at the fine grid (patch_size / decoder_sub), using the decoder_sub
        pipeline to emit decoder_sub^3 distinct vectors per coarse patch.

        Each encoder token is unshuffled into sub^3 sub-tokens, positional embeddings are applied
        at the fine grid, sub blocks refine them, and the result is reshaped to the fine grid.
        Shares decoder_sub's pos embed and blocks (trained during MAE)."""
        assert self.octant_head and self.decoder_sub > 1
        s = self.decoder_sub
        B = images.shape[0]
        assert tuple(images.shape[1:]) == tuple(self.grid_size)
        coarse_grid = [size // self.patch_size for size in self.grid_size]
        fine_grid = [size * s for size in coarse_grid]
        fine_total = fine_grid[0] * fine_grid[1] * fine_grid[2]
        dd = self.decoder_proj.out_features  # decoder_embed_dim
        embeds_fine = images.new_zeros(B, *fine_grid, dd)

        # Encoder + decoder_sub pipeline, one forward per image
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
            h = x[0, self.class_tokens:]  # [M, D]

            # decoder_sub pipeline: unshuffle → pos embed → sub blocks → norm
            h = self.decoder_proj(h)  # [M, Dd]
            M = h.shape[0]
            sub_tokens = self.decoder_unshuffle(h).reshape(M, s**3, -1)  # [M, s³, Dd]
            coarse_coord = self.coord_grid[0, in_mask]  # [M, 3]
            sub_coord = coarse_coord[:, None, :] * s + self.sub_offsets[None, :, :]  # [M, s³, 3]
            sub_tokens = self.decoder_sub_pos_embed(sub_tokens, sub_coord)
            for block in self.decoder_sub_blocks:
                sub_tokens = block(sub_tokens)
            sub_tokens = self.decoder_sub_norm(sub_tokens).to(embeds_fine.dtype)  # [M, s³, Dd]

            # Vectorized scatter: flat indices into fine grid
            flat_idx = (sub_coord[:, :, 0] * fine_grid[1] * fine_grid[2] +
                        sub_coord[:, :, 1] * fine_grid[2] +
                        sub_coord[:, :, 2]).long().flatten()  # [M*s³]
            embeds_fine[ii].flatten(0, 2)[flat_idx] = sub_tokens.flatten(0, 1)

        return embeds_fine

    def global_embed(
        self,
        images: Float[Tensor, "B X Y Z"],
        mask: Float[Tensor, "B X Y Z"],
    ) -> Float[Tensor, "B D"]:
        if self.octant_head:
            # Use dense_embed (routes to octant method) for finer-grained pooling
            embeds = self.dense_embed(images, mask).flatten(1, 3)
            embeds = embeds.mean(dim=1)
        else:
            embeds = self.dense_embed_native(images, mask).flatten(1, 3)
            in_mask = patchify3d(mask > 0, self.patch_size).any(dim=2)
            embeds = (embeds * in_mask[..., None]).sum(dim=1) / in_mask.sum(dim=1, keepdim=True)
        return embeds


# the encoder parts the SiamJEPA teacher copies (update_teacher pairs their parameters by order)
TEACHER_PARTS = ("patch_embed", "pos_embed", "blocks", "norm")


# the frozen eval probes an 8 mm patch grid (eval/probe.py PATCH_GRID_SHAPE, 24x30x24)
EVAL_PATCH_SIZE = 8


def to_eval_grid(
    embeds: Float[Tensor, "B Gx Gy Gz D"],
    in_mask: Bool[Tensor, "B N"],
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


def fine_patch_state_dict(model: "ViTMAE3D", fine: "ViTMAE3D") -> dict[str, Tensor]:
    """P7: weights of a model at patch p with the sub-token decoder (decoder_sub s) for `fine`, the
    same model at patch p / s with the plain decoder, so training can continue on the finer grid.
    - patch_embed: fine.patch_embed(x) == model.patch_embed(x nearest-upsampled to p^3), i.e. each
      s^3 block of the coarse input weights is summed
    - position tables (encoder and decoder input): linear interpolation onto the fine grid,
      align_corners=False, which puts each fine patch centre at its place in the coarse grid
    - decoder_head: the sub-token head already predicts (p / s)^3 voxels, so it is kept with the
      decoder blocks and norm; the sub-token stage (unshuffle, sub position table, sub blocks, sub
      norm) is dropped
    Everything else (class tokens, encoder blocks, global projector) is copied."""
    s, p = model.decoder_sub, fine.patch_size
    assert s > 1 and p * s == model.patch_size and fine.decoder_sub == 1, (model.patch_size, s, p)
    sub_stage = ("decoder_unshuffle.", "decoder_sub_pos_embed.", "decoder_sub_blocks.", "decoder_sub_norm.")
    state = {k: v for k, v in model.state_dict().items() if not k.startswith(sub_stage)}
    weight = state["patch_embed.weight"].reshape(-1, p, s, p, s, p, s)
    state["patch_embed.weight"] = weight.sum(dim=(2, 4, 6)).reshape(-1, p**3)
    for name in ("pos_embed", "decoder_pos_embed"):
        for axis, size in zip("xyz", getattr(fine, name).grid_size):
            table = state[f"{name}.weight_{axis}"]  # (coarse size, D)
            table = F.interpolate(table.T[None], size=size, mode="linear", align_corners=False)
            state[f"{name}.weight_{axis}"] = table[0].T.contiguous()
    return state


def spectral_loss(
    pred: Float[Tensor, "B M V"],
    target: Float[Tensor, "B M V"],
    mask: Float[Tensor, "B M V"],
    highpass: Float[Tensor, "P P Q"],
    patch_size: int,
) -> Float[Tensor, "B M"]:
    """A2 per patch (SL-MAE, arXiv 2606.13315): high-pass weighted squared difference of the log
    Fourier magnitudes of the predicted and target patch, mean over frequency bins.

    Voxels outside the mask are zeroed in both, as in the paper's full-volume version where the
    background is 0. Without this, boundary patches dominate: norm_target turns the background into
    a large constant (e.g. -34) and the masked MSE leaves the predictions there untrained.
    log1p (eps = 1) instead of log(|F| + 1e-6): RMS |F| is ~23 for a unit-variance 8^3 patch, and
    a tiny eps lets noise-level bins and the all-zero initial prediction dominate the loss and its
    gradient (~228 per patch at step 0). FFTs in float32: torch.fft rejects bfloat16."""
    p = patch_size
    shape = (*pred.shape[:-1], p, p, p)
    mask = mask.reshape(shape).float()
    spec_pred = torch.fft.rfftn(pred.float().reshape(shape) * mask, dim=(-3, -2, -1))
    spec_target = torch.fft.rfftn(target.float().reshape(shape) * mask, dim=(-3, -2, -1))
    diff = (torch.log1p(spec_pred.abs()) - torch.log1p(spec_target.abs())).square()
    return (diff * highpass).sum(dim=(-3, -2, -1)) / highpass.numel()


def sigreg(
    z: Float[Tensor, "N D"], num_slices: int = 1024, t_max: float = 3.0, num_points: int = 17
) -> Tensor:
    """LeJEPA's SIGReg: the Epps-Pulley normality statistic of z projected on num_slices random
    unit directions, averaged over directions. Same math as the official lejepa code
    (univariate/epps_pulley.py EppsPulley, multivariate/slicing.py SlicingUnivariateTest):
    N * integral over t in [-3, 3] of |empirical char. function - exp(-t^2/2)|^2 exp(-t^2/2) dt,
    trapezoid rule on 17 points of t >= 0 using the symmetry in t. Small for N(0, I) samples."""
    directions = torch.randn(z.shape[-1], num_slices, device=z.device)
    directions = directions / directions.norm(dim=0)
    x = (z.float() @ directions)[..., None]  # [N, K, 1]
    t = torch.linspace(0, t_max, num_points, device=z.device)
    phi = torch.exp(-0.5 * t**2)
    dt = t_max / (num_points - 1)
    weights = torch.full((num_points,), 2 * dt, device=z.device)
    weights[[0, -1]] = dt
    xt = x * t
    err = (xt.cos().mean(dim=0) - phi).square() + xt.sin().mean(dim=0).square()  # [K, P]
    return (err @ (weights * phi)).mean() * z.shape[0]


def split_subcubes(x: Float[Tensor, "B M P"], patch_size: int, sub: int) -> Float[Tensor, "B M S Q"]:
    """Voxels of each patch (patchify3d order) regrouped into its sub^3 sub-cubes, in
    meshgrid(ij) order of the sub-cube offsets, each in patchify3d order."""
    q = patch_size // sub
    return rearrange(
        x,
        "b m (ox qx oy qy oz qz) -> b m (ox oy oz) (qx qy qz)",
        ox=sub, oy=sub, oz=sub, qx=q, qy=q, qz=q,
    )


def merge_subcubes(x: Float[Tensor, "B M S Q"], patch_size: int, sub: int) -> Float[Tensor, "B M P"]:
    """Inverse of split_subcubes."""
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
