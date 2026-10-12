"""CPU checks of the patch-size plumbing (model.to_eval_grid, dense_embed vs dense_embed_native).

1. A linear field sampled at native patch centres must interpolate exactly onto 8 mm patch centres
   (interior cells; trilinear is exact for linear fields), for patch sizes 12 and 16.
2. Patch size 8 returns dense_embed_native unchanged (the existing models' eval is untouched).
3. Tiny random models at patch 8/12/16 and a readout at 16: shapes of dense_embed (8 mm grid),
   dense_embed_native (own grid), global_embed, and a finite training loss.
"""

import torch

from nanobrain.model import EVAL_PATCH_SIZE, ViTMAE3D, to_eval_grid
from nanobrain.readout import ReadoutViTMAE3D

torch.manual_seed(0)
GRID = (192, 240, 192)

# 1. linear field: value = a . (voxel coordinate of the patch centre) + b
a, b = torch.tensor([0.3, -0.2, 0.5]), 1.0
for p in (12, 16):
    native = [s // p for s in GRID]
    centres = torch.stack(torch.meshgrid(*[torch.arange(n) * p + p / 2 for n in native], indexing="ij"), -1)
    field = (centres @ a + b)[None, ..., None]
    out = to_eval_grid(field, torch.ones(1, field[0, ..., 0].numel(), dtype=torch.bool), p)[0, ..., 0]
    eval_centres = torch.stack(
        torch.meshgrid(*[torch.arange(s // EVAL_PATCH_SIZE) * 8 + 4.0 for s in GRID], indexing="ij"), -1
    )
    expected = eval_centres @ a + b
    k = p // EVAL_PATCH_SIZE + 1  # skip border cells, where interpolation clamps
    inner = (slice(k, -k),) * 3
    err = (out[inner] - expected[inner]).abs().max().item()
    print(f"p{p}: native {tuple(native)} -> {tuple(out.shape)}, interior max error {err:.2e}")
    assert tuple(out.shape) == tuple(s // EVAL_PATCH_SIZE for s in GRID) and err < 1e-4

# 2./3. tiny models on a synthetic head (ellipsoid mask, smooth image)
xyz = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, s) for s in GRID], indexing="ij"), -1)
mask = ((xyz**2).sum(-1) < 0.6).float()[None]
image = (torch.sin(5 * xyz).sum(-1) + 3) [None] * mask
kw = dict(depth=2, embed_dim=32, num_heads=2, decoder_depth=1, decoder_embed_dim=32, decoder_num_heads=2)
for p, n in ((8, 1280), (12, 384), (16, 160)):
    model = ViTMAE3D(grid_size=GRID, patch_size=p, **kw).eval()
    with torch.no_grad():
        native = model.dense_embed_native(image, mask)
        dense = model.dense_embed(image, mask)
        glob = model.global_embed(image, mask)
        loss = model(image, mask, num_visible=n, num_predict=n, num_samples=2, with_state=False, norm_target=True)
    print(f"p{p}: native {tuple(native.shape)}, dense {tuple(dense.shape)}, global {tuple(glob.shape)}, loss {loss.item():.3f}")
    assert tuple(dense.shape[1:4]) == (24, 30, 24) and torch.isfinite(loss)
    if p == EVAL_PATCH_SIZE:
        assert torch.equal(dense, native)
    else:
        assert tuple(native.shape[1:4]) == tuple(s // p for s in GRID)

args = dict(grid_size=GRID, patch_size=16, model_kwargs=kw,
            readout=dict(layers=[1, 2], global_layers=None, combine="cat", global_cls=False))
readout = ReadoutViTMAE3D.from_config(args).eval()
with torch.no_grad():
    native, dense = readout.dense_embed_native(image, mask), readout.dense_embed(image, mask)
print(f"readout p16 cat1-2: native {tuple(native.shape)}, dense {tuple(dense.shape)}")
assert tuple(dense.shape) == (1, 24, 30, 24, 64) and tuple(native.shape) == (1, 12, 15, 12, 64)
print("all checks passed")
