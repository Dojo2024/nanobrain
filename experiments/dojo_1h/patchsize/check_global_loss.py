"""CPU checks of L8 (model.sigreg, model_kwargs.global_loss).

1. sigreg on 1-D samples equals the Epps-Pulley statistic N * int_{-3}^{3} |ecf(t) - exp(-t^2/2)|^2
   exp(-t^2/2) dt, computed independently on a fine grid (the 17-point trapezoid is close, not exact).
2. It is small for N(0, I) samples and large for collapsed or mis-scaled ones.
3. Tiny p16 models with global_loss sigreg / infonce: finite loss, and the global term alone sends
   gradient to the encoder at step 1 (the decoder head starts at zero, the projector does not).
"""

import numpy as np
import torch

from nanobrain.model import ViTMAE3D, sigreg

torch.manual_seed(0)

# 1. vs a fine-grid integral
for name, z in [("normal", torch.randn(64, 1)), ("shifted", torch.randn(64, 1) + 0.7), ("wide", 2 * torch.randn(64, 1))]:
    x = z[:, 0].double().numpy()
    t = np.linspace(-3, 3, 20001)
    ecf = np.exp(1j * np.outer(t, x)).mean(axis=1)
    integrand = np.abs(ecf - np.exp(-t**2 / 2)) ** 2 * np.exp(-t**2 / 2)
    ref = len(x) * np.trapezoid(integrand, t)
    ours = sigreg(z, num_slices=8).item()
    print(f"{name:8} sigreg {ours:.4f}  fine-grid {ref:.4f}  rel diff {abs(ours - ref) / ref:.3f}")
    assert abs(ours - ref) / ref < 0.05

# 2. behaviour in 256-D
normal = sigreg(torch.randn(32, 256)).item()
collapsed = sigreg(torch.randn(1, 256).expand(32, 256)).item()
wide = sigreg(3 * torch.randn(32, 256)).item()
print(f"N(0,I) {normal:.3f}  collapsed {collapsed:.3f}  3x scale {wide:.3f}")
assert normal < collapsed and normal < wide

# 3. tiny p16 models
GRID = (192, 240, 192)
xyz = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, s) for s in GRID], indexing="ij"), -1)
mask = ((xyz**2).sum(-1) < 0.6).float()[None].expand(2, -1, -1, -1).contiguous()
image = (torch.sin(5 * xyz).sum(-1) + 3)[None] * mask
image = image + 0.1 * torch.randn_like(image) * mask
kw = dict(depth=2, embed_dim=32, num_heads=2, decoder_depth=1, decoder_embed_dim=32, decoder_num_heads=2,
          global_hidden_dim=64, global_proj_dim=16, sigreg_slices=64)
for kind in ("sigreg", "infonce"):
    model = ViTMAE3D(grid_size=GRID, patch_size=16, global_loss=kind, decoder_sub=2, **kw)
    loss = model(image, mask, num_visible=168, num_predict=168, num_samples=2, with_state=False, norm_target=True)
    loss.backward()
    grad = model.patch_embed.weight.grad.abs().sum().item()
    print(f"{kind}: loss {loss.item():.4f}, encoder grad at step 1 {grad:.2e}")
    assert torch.isfinite(loss) and grad > 0
print("all checks passed")
