# References:
# capi: https://github.com/facebookresearch/capi/blob/main/model.py
# timm: https://github.com/huggingface/pytorch-image-models/blob/v1.0.20/timm/models/vision_transformer.py

from functools import partial
from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float
from timm.layers import DropPath

Layer = Type[nn.Module]


# Transformer modules adapted from capi (but removed the efficient residual)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        context_dim = context_dim or dim

        # using separate q, k, v weights so that xavier init uses the correct dim.
        # although perhaps technically it should be initialized wrt the head dim..
        # but this is what original mae does.
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

    def extra_repr(self):
        return f"num_heads={self.num_heads}, causal={self.causal}"

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        context: Float[Tensor, "B M D"] | None = None,
    ) -> Float[Tensor, "B N D"]:
        if context is None:
            context = x
        B, N, D = x.shape
        _, M, _ = context.shape
        h = self.num_heads

        q = self.q(x).reshape(B, N, h, D // h).transpose(1, 2)
        k = self.k(context).reshape(B, M, h, D // h).transpose(1, 2)
        v = self.v(context).reshape(B, M, h, D // h).transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: int | float = 4,
        bias: bool = False,
    ) -> None:
        super().__init__()
        hidden_features = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, dim, bias=bias)

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# timm default eps=1e-6
LayerNorm = partial(nn.LayerNorm, eps=1e-6)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        context_dim: int | None = None,
        mlp_ratio: int | float = 4,
        drop_path: float = 0.0,
        norm_layer: Layer = LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            context_dim=context_dim,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            dim=dim,
            mlp_ratio=mlp_ratio,
            bias=proj_bias,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        context: Float[Tensor, "B M D"] | None = None,
    ) -> Float[Tensor, "B N D"]:
        # should the context also be normalized? capi doesn't, so I guess not
        x = x + self.drop_path1(self.attn(self.norm1(x), context=context))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# Hyper-connections: Zhu et al., "Hyper-Connections", ICLR 2025, arXiv 2409.19606.
# Names follow the paper's Algorithm 2. The hidden state is a "hyper hidden matrix" of rate
# streams per token, [B, N, rate, D], instead of one [B, N, D] residual stream.


class HyperConnection(nn.Module):
    """One hyper-connection, wrapping one residual branch (attention or MLP).

    width_connection mixes the streams with the matrix [A_m | A_r] (alpha, [rate, rate+1]): column
    0 (A_m) forms the branch input, columns 1: (A_r) mix the streams with each other.
    depth_connection adds the branch output into each stream, weighted by B (beta, [rate]).
    Dynamic (DHC, eqs. 10-13) adds input-dependent terms tanh(norm(H) W) * s, with W = 0 at init.
    The init (eq. 14) makes the block an ordinary pre-norm residual: B = 1, A_m reads stream
    layer_id % rate, A_r = I.
    """

    def __init__(self, dim: int, rate: int, layer_id: int, dynamic: bool = True) -> None:
        super().__init__()
        self.rate = rate
        self.dynamic = dynamic
        self.static_beta = nn.Parameter(torch.ones(rate))
        init_alpha0 = torch.zeros(rate, 1)
        init_alpha0[layer_id % rate, 0] = 1.0
        self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye(rate)], dim=1))
        if dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, rate + 1))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = LayerNorm(dim)

    def extra_repr(self):
        return f"rate={self.rate}, dynamic={self.dynamic}"

    def width_connection(
        self, h: Float[Tensor, "B N R D"]
    ) -> tuple[Float[Tensor, "B N R+1 D"], Float[Tensor, "... R"]]:
        alpha, beta = self.static_alpha, self.static_beta
        if self.dynamic:
            norm_h = self.layer_norm(h)
            alpha = alpha + torch.tanh(norm_h @ self.dynamic_alpha_fn) * self.dynamic_alpha_scale
            beta = beta + torch.tanh(norm_h @ self.dynamic_beta_fn) * self.dynamic_beta_scale
        # row 0 is the branch input, rows 1: are the mixed streams
        mix_h = alpha.transpose(-1, -2) @ h
        return mix_h, beta

    def depth_connection(
        self,
        mix_h: Float[Tensor, "B N R+1 D"],
        h_o: Float[Tensor, "B N D"],
        beta: Float[Tensor, "... R"],
    ) -> Float[Tensor, "B N R D"]:
        return h_o[..., None, :] * beta[..., :, None] + mix_h[..., 1:, :]


class HyperConnectionBlock(Block):
    """Block with its two residual adds replaced by hyper-connections (the paper's Algorithm 3).
    Takes and returns the hyper hidden matrix [B, N, rate, D]. Each branch counts as one layer
    for the init, so the attention of block i reads stream 2i % rate and its MLP (2i+1) % rate."""

    def __init__(self, *, rate: int, layer_id: int, dynamic: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        dim = kwargs["dim"]
        self.hc_attn = HyperConnection(dim, rate, 2 * layer_id, dynamic)
        self.hc_mlp = HyperConnection(dim, rate, 2 * layer_id + 1, dynamic)

    def forward(
        self,
        h: Float[Tensor, "B N R D"],
        context: Float[Tensor, "B M D"] | None = None,
    ) -> Float[Tensor, "B N R D"]:
        mix_h, beta = self.hc_attn.width_connection(h)
        x = self.drop_path1(self.attn(self.norm1(mix_h[..., 0, :]), context=context))
        h = self.hc_attn.depth_connection(mix_h, x, beta)
        mix_h, beta = self.hc_mlp.width_connection(h)
        x = self.drop_path2(self.mlp(self.norm2(mix_h[..., 0, :])))
        h = self.hc_mlp.depth_connection(mix_h, x, beta)
        return h


# Position embedding


class SeparablePosEmbed3D(nn.Module):
    def __init__(
        self,
        grid_size: tuple[int, int, int],
        embed_dim: int,
        unit: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.unit = unit

        X, Y, Z = grid_size
        self.weight_x = nn.Parameter(torch.empty(X, embed_dim))
        self.weight_y = nn.Parameter(torch.empty(Y, embed_dim))
        self.weight_z = nn.Parameter(torch.empty(Z, embed_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight_x, std=0.02)
        nn.init.trunc_normal_(self.weight_y, std=0.02)
        nn.init.trunc_normal_(self.weight_z, std=0.02)

    def forward(
        self,
        x: Float[Tensor, "B N D"],
        coord: Float[Tensor, "B N 3"],
    ) -> Float[Tensor, "B N D"]:
        pos_ids = torch.floor(coord / self.unit).to(torch.int64)
        x = x + self.weight_x[pos_ids[:, :, 0]]
        x = x + self.weight_y[pos_ids[:, :, 1]]
        x = x + self.weight_z[pos_ids[:, :, 2]]
        return x

    def extra_repr(self):
        return f"{self.embed_dim}, {self.grid_size}, unit={self.unit}"
