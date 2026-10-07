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

    The matrix [A_m | A_r] (alpha, [rate, rate+1]) mixes the streams: column 0 (A_m) forms the
    branch input, columns 1: (A_r) mix the streams with each other. B (beta, [rate]) weights the
    branch output added into each stream. So the block computes H' = A_r^T H + B^T T(A_m^T H).
    Dynamic (DHC, eqs. 10-13) adds input-dependent terms tanh(norm(H) W) * s, with W = 0 at init.
    The init (eq. 14) makes the block an ordinary pre-norm residual: B = 1, A_m reads stream
    layer_id % rate, A_r = I.

    Same math as the paper's Algorithm 2, arranged to move less memory, because at width 512 the
    rate-stream tensors cost more time than the matrix multiplies: W_alpha and W_beta are one fp32
    projection of norm(H), width_connection returns only the branch input A_m^T H, and
    depth_connection forms A_r^T H together with the B-weighted output. The paper's mix_h
    ([rate+1] streams) is never stored, and the stream sums are short loops torch.compile fuses.
    """

    def __init__(
        self,
        dim: int,
        rate: int,
        layer_id: int,
        dynamic: bool = True,
        stream_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.rate = rate
        self.dynamic = dynamic
        # dtype the streams are stored in between branches, None keeps the computed fp32
        self.stream_dtype = stream_dtype
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
    ) -> tuple[Float[Tensor, "B N D"], Float[Tensor, "... R R+1"], Float[Tensor, "... R"]]:
        alpha, beta = self.static_alpha, self.static_beta
        if self.dynamic:
            with torch.autocast(h.device.type, enabled=False):
                fn = torch.cat([self.dynamic_alpha_fn, self.dynamic_beta_fn[:, None]], dim=1)
                norm_h = self.layer_norm(h.to(self.layer_norm.weight.dtype))
                dynamic = torch.tanh(norm_h @ fn)
            alpha = alpha + dynamic[..., :-1] * self.dynamic_alpha_scale
            beta = beta + dynamic[..., -1] * self.dynamic_beta_scale
        # branch input A_m^T H
        x = sum(alpha[..., r, 0, None] * h[..., r, :] for r in range(self.rate))
        return x, alpha, beta

    def depth_connection(
        self,
        h: Float[Tensor, "B N R D"],
        h_o: Float[Tensor, "B N D"],
        alpha: Float[Tensor, "... R R+1"],
        beta: Float[Tensor, "... R"],
    ) -> Float[Tensor, "B N R D"]:
        # A_r^T H + B^T h_o
        mixed = sum(alpha[..., r, 1:, None] * h[..., r, None, :] for r in range(self.rate))
        h = mixed + beta[..., :, None] * h_o[..., None, :]
        return h if self.stream_dtype is None else h.to(self.stream_dtype)


class HyperConnectionBlock(Block):
    """Block with its two residual adds replaced by hyper-connections (the paper's Algorithm 3).
    Takes and returns the hyper hidden matrix [B, N, rate, D]. Each branch counts as one layer
    for the init, so the attention of block i reads stream 2i % rate and its MLP (2i+1) % rate."""

    def __init__(
        self,
        *,
        rate: int,
        layer_id: int,
        dynamic: bool = True,
        stream_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        dim = kwargs["dim"]
        self.hc_attn = HyperConnection(dim, rate, 2 * layer_id, dynamic, stream_dtype)
        self.hc_mlp = HyperConnection(dim, rate, 2 * layer_id + 1, dynamic, stream_dtype)

    def forward(
        self,
        h: Float[Tensor, "B N R D"],
        context: Float[Tensor, "B M D"] | None = None,
    ) -> Float[Tensor, "B N R D"]:
        x, alpha, beta = self.hc_attn.width_connection(h)
        x = self.drop_path1(self.attn(self.norm1(x), context=context))
        h = self.hc_attn.depth_connection(h, x, alpha, beta)
        x, alpha, beta = self.hc_mlp.width_connection(h)
        x = self.drop_path2(self.mlp(self.norm2(x)))
        h = self.hc_mlp.depth_connection(h, x, alpha, beta)
        return h


# Frac-connections: Zhu et al., "Frac-Connections: Fractional Extension of Hyper-Connections",
# arXiv 2503.14125. Names follow the paper's Algorithm 2.


class FracConnection(nn.Module):
    """One frac-connection, wrapping one residual branch (attention or MLP).

    Instead of copying the hidden state into rate streams (HyperConnection), it splits it into
    rate fractions of D / rate (eq. 6), so the hidden state stays [B, N, D] and costs no extra
    memory. alpha = [Y | A] ([rate, 2 rate]): Y mixes the fractions into the branch input, A mixes
    them with each other; beta (B, [rate]) scales each fraction of the branch output. So the block
    computes H' = A^T H + B * T(Y^T H), with H the hidden state viewed as [rate, D / rate] (eq. 8).
    Dynamic (DFC, eqs. 11-14) adds tanh(norm(H) W) * s per fraction, with W = 0 at init. The init
    (eq. 15), Y = A = I and B = 1, makes the block an ordinary pre-norm residual.

    Arranged like HyperConnection: Y^T H is formed in width_connection and A^T H in
    depth_connection, so the paper's mix_h is never stored. The paper casts mix_h to bf16; here
    the hidden state stays fp32, like the plain residual.
    """

    def __init__(
        self,
        dim: int,
        rate: int,
        dynamic: bool = True,
        stream_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        assert dim % rate == 0, (dim, rate)
        self.rate = rate
        self.dynamic = dynamic
        # dtype the hidden state is stored in between branches, None keeps the computed fp32
        self.stream_dtype = stream_dtype
        self.static_beta = nn.Parameter(torch.ones(rate))
        self.static_alpha = nn.Parameter(torch.cat([torch.eye(rate), torch.eye(rate)], dim=1))
        if dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim // rate, 2 * rate))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim // rate))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = LayerNorm(dim // rate)

    def extra_repr(self):
        return f"rate={self.rate}, dynamic={self.dynamic}"

    def width_connection(
        self, h: Float[Tensor, "B N D"]
    ) -> tuple[Float[Tensor, "B N D"], Float[Tensor, "... R 2R"], Float[Tensor, "... R"]]:
        h = h.unflatten(-1, (self.rate, -1))
        alpha, beta = self.static_alpha, self.static_beta
        if self.dynamic:
            with torch.autocast(h.device.type, enabled=False):
                fn = torch.cat([self.dynamic_alpha_fn, self.dynamic_beta_fn[:, None]], dim=1)
                norm_h = self.layer_norm(h.to(self.layer_norm.weight.dtype))
                dynamic = torch.tanh(norm_h @ fn)
            alpha = alpha + dynamic[..., :-1] * self.dynamic_alpha_scale
            beta = beta + dynamic[..., -1] * self.dynamic_beta_scale
        # branch input Y^T H
        x = sum(alpha[..., i, : self.rate, None] * h[..., i, None, :] for i in range(self.rate))
        return x.flatten(-2), alpha, beta

    def depth_connection(
        self,
        h: Float[Tensor, "B N D"],
        h_o: Float[Tensor, "B N D"],
        alpha: Float[Tensor, "... R 2R"],
        beta: Float[Tensor, "... R"],
    ) -> Float[Tensor, "B N D"]:
        # A^T H + B * h_o, per fraction
        h = h.unflatten(-1, (self.rate, -1))
        mixed = sum(alpha[..., i, self.rate :, None] * h[..., i, None, :] for i in range(self.rate))
        h = (mixed + beta[..., None] * h_o.unflatten(-1, (self.rate, -1))).flatten(-2)
        return h if self.stream_dtype is None else h.to(self.stream_dtype)


class FracConnectionBlock(Block):
    """Block with its two residual adds replaced by frac-connections (the paper's Algorithm 3).
    Takes and returns the usual hidden state [B, N, D]."""

    def __init__(
        self,
        *,
        rate: int,
        dynamic: bool = True,
        stream_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hc_attn = FracConnection(kwargs["dim"], rate, dynamic, stream_dtype)
        self.hc_mlp = FracConnection(kwargs["dim"], rate, dynamic, stream_dtype)

    def forward(
        self,
        h: Float[Tensor, "B N D"],
        context: Float[Tensor, "B M D"] | None = None,
    ) -> Float[Tensor, "B N D"]:
        x, alpha, beta = self.hc_attn.width_connection(h)
        x = self.drop_path1(self.attn(self.norm1(x), context=context))
        h = self.hc_attn.depth_connection(h, x, alpha, beta)
        x, alpha, beta = self.hc_mlp.width_connection(h)
        x = self.drop_path2(self.mlp(self.norm2(x)))
        h = self.hc_mlp.depth_connection(h, x, alpha, beta)
        return h


def sinkhorn_knopp(logits: Tensor, iters: int) -> Tensor:
    """Doubly stochastic matrix from logits [..., n, n] (mHC eq. 9): M0 = exp(logits), then iters
    rounds of M = T_r(T_c(M)), columns normalized then rows. Subtracting the max first changes
    nothing (both normalizations are scale invariant) and keeps exp finite."""
    m = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
    for _ in range(iters):
        m = m / m.sum(dim=-2, keepdim=True)
        m = m / m.sum(dim=-1, keepdim=True)
    return m


class ManifoldHyperConnection(nn.Module):
    """One manifold-constrained hyper-connection (mHC, Xie et al., arXiv 2512.24880), wrapping one
    residual branch. With the hidden matrix x_l [n, C] the block computes (eq. 3)

        x_{l+1} = H_res x_l + H_post^T F(H_pre x_l)

    with coefficients from the flattened, RMS-normed hidden matrix (eqs. 7-8):

        x' = RMSNorm(vec(x_l))                          [1, nC]
        H_pre  = sigmoid(a_pre * x' phi_pre + b_pre)      [1, n]
        H_post = 2 sigmoid(a_post * x' phi_post + b_post) [1, n]
        H_res  = SinkhornKnopp(a_res * mat(x' phi_res) + b_res)  [n, n], doubly stochastic

    The paper gives a = 0.01 but not the init of phi or b. Here phi = 0, as HC's dynamic
    projections, and b encodes HC's init: the branch reads mostly stream layer_id % n (H_pre
    .982 there, .006 on the others, summing to 1), writes every stream (b_post 0, so H_post = 1)
    and leaves the streams unmixed (b_res = 4 I, so H_res has .95 on the diagonal). As with HC,
    an n-stream encoder that starts from copies of the embedding is then exactly a pre-norm
    residual at init: identical streams stay identical under any doubly stochastic H_res and
    equal writes, and H_pre x_l is their weighted mean, the stream itself. The coefficients are computed in fp32 with autocast off.
    """

    def __init__(
        self,
        dim: int,
        rate: int,
        layer_id: int,
        sinkhorn_iters: int = 20,
        stream_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.rate = rate
        self.sinkhorn_iters = sinkhorn_iters
        self.stream_dtype = stream_dtype
        self.rms_norm = nn.RMSNorm(rate * dim, eps=1e-6)
        # phi_pre | phi_post | phi_res as one projection
        self.dynamic_fn = nn.Parameter(torch.zeros(rate * dim, 2 * rate + rate * rate))
        self.pre_scale = nn.Parameter(torch.full((1,), 0.01))
        self.post_scale = nn.Parameter(torch.full((1,), 0.01))
        self.res_scale = nn.Parameter(torch.full((1,), 0.01))
        # sigmoid(b_pre) is .982 on stream layer_id % n and sums to exactly 1 over the streams
        rest = (1 - torch.sigmoid(torch.tensor(4.0))) / (rate - 1)
        static_pre = torch.full((rate,), torch.logit(rest).item())
        static_pre[layer_id % rate] = 4.0
        self.static_pre = nn.Parameter(static_pre)
        self.static_post = nn.Parameter(torch.zeros(rate))
        self.static_res = nn.Parameter(4.0 * torch.eye(rate))

    def extra_repr(self):
        return f"rate={self.rate}, sinkhorn_iters={self.sinkhorn_iters}"

    def width_connection(
        self, h: Float[Tensor, "... R D"]
    ) -> tuple[Float[Tensor, "... D"], Float[Tensor, "... R"], Float[Tensor, "... R R"]]:
        """Branch input H_pre x_l, plus H_post and H_res for depth_connection."""
        n = self.rate
        with torch.autocast(h.device.type, enabled=False):
            h = h.to(self.rms_norm.weight.dtype)
            z = self.rms_norm(h.flatten(-2)) @ self.dynamic_fn
            pre = torch.sigmoid(self.pre_scale * z[..., :n] + self.static_pre)
            post = 2 * torch.sigmoid(self.post_scale * z[..., n : 2 * n] + self.static_post)
            res = self.res_scale * z[..., 2 * n :].unflatten(-1, (n, n)) + self.static_res
            res = sinkhorn_knopp(res, self.sinkhorn_iters)
            x = sum(pre[..., r, None] * h[..., r, :] for r in range(n))
        return x, post, res

    def depth_connection(
        self,
        h: Float[Tensor, "... R D"],
        h_o: Float[Tensor, "... D"],
        post: Float[Tensor, "... R"],
        res: Float[Tensor, "... R R"],
    ) -> Float[Tensor, "... R D"]:
        """H_res x_l + H_post^T h_o: stream i gets sum_j H_res[i, j] x_j + H_post[i] h_o."""
        out = post[..., :, None] * h_o[..., None, :]
        for r in range(self.rate):
            out = out + res[..., :, r, None] * h[..., r, None, :]
        return out if self.stream_dtype is None else out.to(self.stream_dtype)


class ManifoldHyperConnectionBlock(Block):
    """Block with its two residual adds replaced by mHC. Takes and returns the hidden matrix
    [B, N, n, D]. Each branch is one layer for the init, as in HyperConnectionBlock."""

    def __init__(
        self,
        *,
        rate: int,
        layer_id: int,
        stream_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hc_attn = ManifoldHyperConnection(kwargs["dim"], rate, 2 * layer_id, stream_dtype=stream_dtype)
        self.hc_mlp = ManifoldHyperConnection(kwargs["dim"], rate, 2 * layer_id + 1, stream_dtype=stream_dtype)

    def forward(
        self,
        h: Float[Tensor, "B N R D"],
        context: Float[Tensor, "B M D"] | None = None,
    ) -> Float[Tensor, "B N R D"]:
        x, post, res = self.hc_attn.width_connection(h)
        x = self.drop_path1(self.attn(self.norm1(x), context=context))
        h = self.hc_attn.depth_connection(h, x, post, res)
        x, post, res = self.hc_mlp.width_connection(h)
        x = self.drop_path2(self.mlp(self.norm2(x)))
        h = self.hc_mlp.depth_connection(h, x, post, res)
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
