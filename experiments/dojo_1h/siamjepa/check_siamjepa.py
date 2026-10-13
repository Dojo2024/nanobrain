"""CPU checks of SiamJEPA (model.forward_siamjepa, update_teacher; train.make_optimizer).
1. The teacher starts equal to the student encoder, has no gradient and no optimizer slot; the
   SiamJEPA heads go to AdamW, the encoder blocks to Muon.
2. A training step: finite loss, gradients reach the encoder, h, the projector, posterior, prior,
   Z embedding and decoder; update_teacher(m) gives m * teacher + (1 - m) * student exactly.
3. The KL term: loss(kl_scale 0.01) - loss(kl_scale 0) = 0.01 * KL with KL >= the 0.1 free bits.
4. Random Shuffle Teacher changes the targets (higher loss at step 0 for the same draws).
5. The checkpoint round trip rebuilds the teacher; eval's dense_embed / global_embed run the student;
   a default (MAE) model has no new state dict keys.
"""

import torch
from omegaconf import OmegaConf

from nanobrain.model import TEACHER_PARTS, ViTMAE3D
from nanobrain.train import make_optimizer

torch.manual_seed(0)
GRID = (192, 240, 192)
args = OmegaConf.load("src/nanobrain/config/default_pretrain.yaml")
args.merge_with(OmegaConf.from_dotlist([
    "patch_size=16", "num_visible=96", "num_predict=96", "optimizer=muon", "muon_impl=batched",
    "fused_adamw=false", "lr=0.001", "model_kwargs.objective=siamjepa",
    "model_kwargs.depth=2", "model_kwargs.embed_dim=32", "model_kwargs.num_heads=2",
    "model_kwargs.decoder_depth=1", "model_kwargs.decoder_embed_dim=32", "model_kwargs.decoder_num_heads=2",
]))
xyz = torch.stack(torch.meshgrid(*[torch.linspace(-1, 1, s) for s in GRID], indexing="ij"), -1)
mask = ((xyz**2).sum(-1) < 0.6).float()[None].repeat(4, 1, 1, 1)
image = (torch.sin(5 * xyz).sum(-1) + 3)[None] * mask + 0.1 * torch.randn_like(mask) * mask


def loss_of(model, seed, **kw):
    torch.manual_seed(seed)
    return model(image, mask, num_visible=args.num_visible, num_predict=args.num_predict,
                 with_state=False, **kw)


model = ViTMAE3D.from_config(args)
model.train()

# 1
student = [p for name in TEACHER_PARTS for p in getattr(model, name).parameters()] + [model.cls_token]
teacher = list(model.teacher.parameters()) + [model.teacher_cls_token]
assert all(torch.equal(s, t) for s, t in zip(student, teacher)) and len(student) == len(teacher)
assert not any(t.requires_grad for t in teacher)
optimizer = make_optimizer(args, model)
in_opt = {id(p) for g in optimizer.muon.param_groups + optimizer.adamw.param_groups for p in g["params"]}
assert not any(id(t) in in_opt for t in teacher)
muon_ids = {id(p) for g in optimizer.muon.param_groups for p in g["params"]}
assert any(id(p) in muon_ids for p in model.blocks.parameters())
assert id(model.siam_posterior[0].weight) in in_opt and id(model.siam_posterior[0].weight) not in muon_ids
print("1. teacher = student copy, frozen, not in the optimizer; SiamJEPA heads on AdamW: ok")

# 2
loss = loss_of(model, 1)
loss.backward()
assert torch.isfinite(loss)
for name in ("blocks.0", "siam_h", "siam_projector", "siam_posterior", "siam_prior", "siam_z_embed",
             "decoder_blocks.0", "decoder_head", "patch_embed"):
    grads = [p.grad for n, p in model.named_parameters() if n.startswith(name)]
    assert grads and all(g is not None for g in grads) and sum(g.abs().sum() for g in grads) > 0, name
optimizer.step()
old_teacher = [t.clone() for t in teacher]
model.update_teacher(0.9)
for t, t0, s in zip(teacher, old_teacher, student):
    torch.testing.assert_close(t, 0.9 * t0 + 0.1 * s)
print(f"2. step: loss {loss.item():.3f}, gradients reach every trained part; EMA update exact: ok")

# 3
model.eval()  # BatchNorm running stats fixed so the two calls see the same network
with torch.no_grad():
    l0 = loss_of(model, 2, siam_kl_scale=0.0)
    l1 = loss_of(model, 2, siam_kl_scale=0.01)
kl = (l1 - l0) / 0.01
assert kl >= 0.1 - 1e-4, kl
print(f"3. KL term {kl.item():.3f} (>= 0.1 free bits), added with weight siam_kl_scale: ok")

# 4
fresh = ViTMAE3D.from_config(args).eval()
with torch.no_grad():
    plain = loss_of(fresh, 3)
    shuffled = loss_of(fresh, 3, shuffle_teacher=True)
assert not torch.isclose(plain, shuffled), (plain, shuffled)
print(f"4. Random Shuffle Teacher changes the targets: loss {plain.item():.3f} -> {shuffled.item():.3f}: ok")

# 5
rebuilt = ViTMAE3D.from_config(OmegaConf.to_container(args))
rebuilt.load_state_dict(model.state_dict())
torch.testing.assert_close(rebuilt.teacher_cls_token, model.teacher_cls_token)
rebuilt.eval()
with torch.no_grad():
    dense = rebuilt.dense_embed(image[:1], mask[:1])
    glob = rebuilt.global_embed(image[:1], mask[:1])
assert dense.shape == (1, 24, 30, 24, 32) and glob.shape == (1, 32), (dense.shape, glob.shape)
mae_args = OmegaConf.load("src/nanobrain/config/default_pretrain.yaml")
mae_keys = set(ViTMAE3D.from_config(mae_args).state_dict())
assert not any(k.startswith(("teacher", "siam_")) for k in mae_keys)
print(f"5. checkpoint round trip, eval embeddings {tuple(dense.shape)} / {tuple(glob.shape)}, MAE keys unchanged: ok")
