"""CPU checks of A2 (model.spectral_loss, spectral_weight).
1. A prediction that is exact inside the mask scores 0 in boundary patches too: the voxels outside the
   mask, which the masked MSE never trains, do not enter the loss.
2. Size and gradient at the all-zero initial prediction (zero-init decoder head) and near the target.
3. The rfftn layout: the loss is invariant to a circular shift of the patch (magnitudes only), and the
   reshape keeps patchify3d's (px py pz) voxel order (a shift along z moves voxels by 1).
4. Tiny p8 and p16 models with spectral_weight 0.1: finite loss, gradients reach the encoder, and the
   default model's state dict has no new keys (spectral_mask is not persistent), so old checkpoints load.
"""

import torch

from nanobrain.model import ViTMAE3D, patchify3d, spectral_loss

torch.manual_seed(0)
GRID = (192, 240, 192)
xyz = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, s) for s in GRID], indexing="ij"), -1)
mask = ((xyz**2).sum(-1) < 0.6).float()[None]
image = (torch.sin(5 * xyz).sum(-1) + 3)[None] * mask
kw = dict(depth=2, embed_dim=32, num_heads=2, decoder_depth=1, decoder_embed_dim=32, decoder_num_heads=2)
highpass = ViTMAE3D(grid_size=GRID, patch_size=8, **kw).spectral_mask

# norm_target as in ViTMAE3D.forward
p = 8
x, m = patchify3d(image, p), patchify3d(mask, p)
count = m.sum(-1, keepdim=True).clamp_min(1)
mean = (m * x).sum(-1, keepdim=True) / count
var = (m * (x - mean) ** 2).sum(-1, keepdim=True) / count
t = (x - mean) / (var + 1e-6).sqrt()
full, part = m.sum(-1) == p**3, (m.sum(-1) > 0) & (m.sum(-1) < p**3)

# 1
exact_inside = t * m + 5 * torch.randn_like(t) * (1 - m)
loss = spectral_loss(exact_inside, t, m, highpass, p)
assert loss[part].abs().max() < 1e-4 and loss[full].abs().max() < 1e-4, loss[part].max()
print(f"1. exact inside the mask: max loss {loss[part | full].max():.1e} over {int(part.sum())} boundary "
      f"and {int(full.sum())} full patches: ok")

# 2
tf, mf = t[full][:512], m[full][:512]
for name, pred in [("zeros (step 0)", torch.zeros_like(tf)), ("target + 0.3 noise", tf + 0.3 * torch.randn_like(tf)),
                   ("0.7 x target (blurry)", 0.7 * tf)]:
    pred = pred.clone().requires_grad_(True)
    loss = spectral_loss(pred, tf, mf, highpass, p).mean()
    (grad,) = torch.autograd.grad(loss, pred)
    mse_grad = (2 * (pred - tf) / pred.numel()).abs().mean()
    print(f"2. {name:22s} loss {loss.item():.3f}  mean |grad| {grad.abs().mean():.1e} (MSE {mse_grad:.1e})  "
          f"max {grad.abs().max():.1e}")

# 3
cube = tf[:4].reshape(4, p, p, p)
shifted = torch.roll(cube, shifts=3, dims=-1).reshape(4, -1)
assert spectral_loss(shifted, tf[:4], torch.ones_like(shifted), highpass, p).abs().max() < 1e-4
assert torch.equal(torch.roll(cube, 1, dims=-1).reshape(4, -1)[:, 1], cube.reshape(4, -1)[:, 0])
print("3. circular-shift invariant, (px py pz) order kept: ok")

# 4
default_keys = set(ViTMAE3D(grid_size=GRID, patch_size=8, **kw).state_dict())
for patch, nv in ((8, 256), (16, 168)):
    extra = dict(decoder_sub=2, decoder_sub_depth=1) if patch == 16 else {}
    model = ViTMAE3D(grid_size=GRID, patch_size=patch, **kw, **extra)
    assert "spectral_mask" not in model.state_dict()
    torch.nn.init.normal_(model.decoder_head.weight, std=0.02)  # zero init would block the encoder gradient
    torch.manual_seed(1)
    base, _ = model(image, mask, num_visible=nv, num_predict=nv, num_samples=2, norm_target=True)
    torch.manual_seed(1)
    loss, _ = model(image, mask, num_visible=nv, num_predict=nv, num_samples=2, norm_target=True, spectral_weight=0.1)
    loss.backward()
    enc = sum(q.grad.abs().sum() for q in model.blocks[0].parameters() if q.grad is not None)
    assert torch.isfinite(loss) and enc > 0
    print(f"4. p{patch}: loss {base.item():.3f} -> {loss.item():.3f} with spectral_weight 0.1, encoder grad {enc:.1e}: ok")
assert set(ViTMAE3D(grid_size=GRID, patch_size=8, **kw).state_dict()) == default_keys
print("4. default state dict keys unchanged: ok")
