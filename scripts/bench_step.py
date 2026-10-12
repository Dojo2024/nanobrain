"""Time one training step (forward + backward + optimizer) and peak GPU memory, on synthetic
scans so data loading is out of the picture. Same model setup as train.py: bf16 autocast,
model.compile(), MuonAdamW batched.

usage: python scripts/bench_step.py <label> [model_kwargs overrides as key=value ...]
    e.g. python scripts/bench_step.py dhc4 hyper_connections=4 hc_dynamic=true
    label "naive-..." swaps in the first hyper-connections version (4203b28), for comparison.
"""

import sys
import time

import torch
import torch.nn.functional as F
import yaml

from nanobrain import modules
from nanobrain.model import ViTMAE3D
from nanobrain.utils import misc


class NaiveHyperConnection(modules.HyperConnection):
    """The paper's Algorithm 2 as first committed (4203b28): stores mix_h, rate+1 streams."""

    def width_connection(self, h):
        alpha, beta = self.static_alpha, self.static_beta
        if self.dynamic:
            norm_h = self.layer_norm(h)
            alpha = alpha + torch.tanh(norm_h @ self.dynamic_alpha_fn) * self.dynamic_alpha_scale
            beta = beta + torch.tanh(norm_h @ self.dynamic_beta_fn) * self.dynamic_beta_scale
        return alpha.transpose(-1, -2) @ h, beta

    def depth_connection(self, mix_h, h_o, beta):
        return h_o[..., None, :] * beta[..., :, None] + mix_h[..., 1:, :]


def naive_block_forward(self, h, context=None):
    mix_h, beta = self.hc_attn.width_connection(h)
    x = self.drop_path1(self.attn(self.norm1(mix_h[..., 0, :]), context=context))
    h = self.hc_attn.depth_connection(mix_h, x, beta)
    mix_h, beta = self.hc_mlp.width_connection(h)
    x = self.drop_path2(self.mlp(self.norm2(mix_h[..., 0, :])))
    return self.hc_mlp.depth_connection(mix_h, x, beta)


def main():
    label, overrides = sys.argv[1], sys.argv[2:]
    if label.startswith("naive"):
        modules.HyperConnection.width_connection = NaiveHyperConnection.width_connection
        modules.HyperConnection.depth_connection = NaiveHyperConnection.depth_connection
        modules.HyperConnectionBlock.forward = naive_block_forward

    model_kwargs = dict(embed_dim=512, num_heads=8)
    for item in overrides:
        key, value = item.split("=")
        model_kwargs[key] = yaml.safe_load(value)
    torch.manual_seed(0)
    model = ViTMAE3D(grid_size=(192, 240, 192), patch_size=8, **model_kwargs).cuda()
    model.compile()
    groups = misc.get_param_groups(model, muon=True)
    optimizer = misc.MuonAdamW(groups, betas=(0.9, 0.95), fused=True, impl="batched")
    misc.update_lr(optimizer.param_groups, 1e-4)
    misc.update_wd(optimizer.param_groups, 0.05)

    images = torch.randn(32, 192, 240, 192, device="cuda")
    masks = torch.ones(32, 192, 240, 192, device="cuda")

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(images, masks, num_visible=1280, num_predict=1280, num_samples=2,
                         with_state=False, norm_target=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss

    for _ in range(15):  # compile and warm up
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    num_steps = 40
    start = time.perf_counter()
    for _ in range(num_steps):
        loss = step()
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / num_steps
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"RESULT {label:14s} {seconds:.4f} s/step  {32 / seconds:6.1f} vol/s  "
          f"peak {peak:5.1f} GiB  loss {loss.item():.4f}", flush=True)


if __name__ == "__main__":
    main()
