"""CPU checks of P7 (train.switch_patch_size, model.fine_patch_state_dict).
1. The converted patch embedding of an 8^3 patch equals the 16 mm embedding of that patch
   nearest-upsampled to 16^3.
2. Position tables: a table linear in the coarse index stays linear at the fine patch centres
   (fine index j sits at coarse index (j + 0.5) / 2 - 0.5), away from the clamped ends.
3. The train.py path on a tiny model: Muon steps at p16 with the sub-token decoder, the switch, a new
   optimizer, Muon steps at p8; losses finite; args now describe the p8 model, so a checkpoint's
   args rebuild it and load its state dict strictly (as the eval does). Untouched weights are copied.
"""

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from nanobrain.model import ViTMAE3D
from nanobrain.train import make_optimizer, switch_patch_size

torch.manual_seed(0)
GRID = (192, 240, 192)
args = OmegaConf.load("src/nanobrain/config/default_pretrain.yaml")
args.merge_with(OmegaConf.from_dotlist([
    "patch_size=16", "num_visible=168", "num_predict=168", "norm_target=true", "optimizer=muon",
    "muon_impl=batched", "fused_adamw=false", "lr=0.001", "switch_min=30",
    "model_kwargs.depth=2", "model_kwargs.embed_dim=32", "model_kwargs.num_heads=2",
    "model_kwargs.decoder_depth=1", "model_kwargs.decoder_embed_dim=32", "model_kwargs.decoder_num_heads=2",
    "model_kwargs.decoder_sub=2", "model_kwargs.decoder_sub_depth=1",
]))
xyz = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, s) for s in GRID], indexing="ij"), -1)
mask = ((xyz**2).sum(-1) < 0.6).float()[None]
image = (torch.sin(5 * xyz).sum(-1) + 3)[None] * mask


def steps(model, optimizer, n=2):
    for _ in range(n):
        loss, _ = model(image, mask, num_visible=args.num_visible, num_predict=args.num_predict,
                        num_samples=2, norm_target=True)
        assert torch.isfinite(loss), loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


coarse = ViTMAE3D.from_config(args)
torch.nn.init.normal_(coarse.decoder_head.weight, std=0.02)
loss16 = steps(coarse, make_optimizer(args, coarse))
before = {k: v.clone() for k, v in coarse.state_dict().items()}

fine = switch_patch_size(args, coarse, torch.device("cpu"))

# 1
x = torch.randn(4, 8, 8, 8)
up = x.repeat_interleave(2, 1).repeat_interleave(2, 2).repeat_interleave(2, 3)
torch.testing.assert_close(fine.patch_embed(x.reshape(4, -1)), coarse.patch_embed(up.reshape(4, -1)))
print("1. patch_embed(8^3 patch) == coarse patch_embed(upsampled patch): ok")

# 2
coarse_table = torch.arange(12.0)[:, None] * torch.tensor([[1.0, -2.0]])
fine_table = F.interpolate(coarse_table.T[None], size=24, mode="linear", align_corners=False)[0].T
centres = (torch.arange(24.0) + 0.5) / 2 - 0.5
torch.testing.assert_close(fine_table[1:-1, 0], centres[1:-1])
assert tuple(fine.pos_embed.weight_y.shape) == (30, 32) and tuple(fine.decoder_pos_embed.weight_z.shape) == (24, 32)
print("2. position tables interpolated to fine patch centres, shapes 24x30x24: ok")

# 3
assert (args.patch_size, args.num_visible, args.num_predict, args.model_kwargs.decoder_sub) == (8, 1280, 1280, 1)
for key in ("cls_token", "blocks.1.mlp.fc2.weight", "norm.weight", "decoder_proj.weight", "mask_token",
            "decoder_blocks.0.attn.proj.weight", "decoder_head.weight"):
    assert key in before, key
    torch.testing.assert_close(fine.state_dict()[key], before[key])
assert not any(k.startswith("decoder_sub") for k in fine.state_dict())
loss8 = steps(fine, make_optimizer(args, fine))
rebuilt = ViTMAE3D.from_config(OmegaConf.to_container(args))
rebuilt.load_state_dict(fine.state_dict())
print(f"3. p16 loss {loss16:.3f} -> switch -> p8 loss {loss8:.3f}; args rebuild the p8 model; copies ok")
