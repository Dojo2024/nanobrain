"""CPU checks of P14 (model.decoder_sub, model.layout_head).

1. split_subcubes at patch 16 regroups each parent patch's voxels exactly as patchify3d at 8 does
   for its 8 child cubes (child index 2 * parent + offset), and merge_subcubes inverts it.
2. Tiny p16 models with decoder_sub 2 (depth 0 and 1) and with layout_head: finite loss, gradients
   reach the encoder, predictions keep the parent patch shape.
3. The default model (decoder_sub 1, no layout head) has the same parameters as before, so old
   checkpoints still load.
"""

import torch

from nanobrain.model import ViTMAE3D, merge_subcubes, patchify3d, split_subcubes

torch.manual_seed(0)

# 1. sub-cube regrouping vs patchify3d at 8
vol = torch.randn(1, 32, 48, 32)
p16 = patchify3d(vol, 16)  # parents on a 2x3x2 grid
p8 = patchify3d(vol, 8).reshape(1, 4, 6, 4, 512)
sub = split_subcubes(p16, 16, 2)
offsets = torch.stack(torch.meshgrid(*[torch.arange(2)] * 3, indexing="ij"), -1).reshape(-1, 3)
parents = torch.stack(torch.meshgrid(torch.arange(2), torch.arange(3), torch.arange(2), indexing="ij"), -1).reshape(-1, 3)
for m, (i, j, k) in enumerate(parents.tolist()):
    for o, (a, b, c) in enumerate(offsets.tolist()):
        assert torch.equal(sub[0, m, o], p8[0, 2 * i + a, 2 * j + b, 2 * k + c])
assert torch.equal(merge_subcubes(sub, 16, 2), p16)
print("split/merge match patchify3d at 8: ok")

# 2. tiny p16 models
GRID = (192, 240, 192)
xyz = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, s) for s in GRID], indexing="ij"), -1)
mask = ((xyz**2).sum(-1) < 0.6).float()[None]
image = (torch.sin(5 * xyz).sum(-1) + 3)[None] * mask
kw = dict(depth=2, embed_dim=32, num_heads=2, decoder_depth=1, decoder_embed_dim=32, decoder_num_heads=2)
for extra in (dict(decoder_sub=2, decoder_sub_depth=1), dict(decoder_sub=2, decoder_sub_depth=0), dict(layout_head=True)):
    model = ViTMAE3D(grid_size=GRID, patch_size=16, **kw, **extra)
    # the decoder head starts at zero (no hockey stick), so the encoder gets gradient from step 2
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(2):
        opt.zero_grad()
        loss, state = model(image, mask, num_visible=168, num_predict=168, num_samples=2, norm_target=True)
        loss.backward()
        opt.step()
    grad = model.patch_embed.weight.grad.abs().sum().item()
    print(f"{extra}: loss {loss.item():.4f}, pred {tuple(state['pred_patches'].shape)}, encoder grad {grad:.2e}")
    assert torch.isfinite(loss) and grad > 0 and state["pred_patches"].shape[-1] == 16**3

# 3. default parameters unchanged
default = set(ViTMAE3D(grid_size=GRID, patch_size=8, **kw).state_dict())
assert not any("sub" in k for k in default), "default model gained parameters"
print("default model parameters unchanged: ok")
