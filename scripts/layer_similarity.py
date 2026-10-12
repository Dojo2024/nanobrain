"""Layer similarity of trained encoders, plain vs hyper-connections, after Zhu et al. 2024 Fig. 3.

For each checkpoint, on the same pretraining volumes, embedded whole as the frozen eval does
(no masking, one image per forward, bf16 autocast), per token:
- cos_sub: cosine between the inputs of consecutive sublayers. Each attention and MLP counts as a
  layer, as in the HC init. A sublayer's input is what its norm receives: the residual stream for
  plain, the mixed branch input h_0 = A_m^T H for HC (the paper's Fig. 3 quantity).
- update: ||F(x)|| / ||x|| per sublayer, F the branch output, x its input.
- cos_block, delta: cosine and relative change ||S_l - S_{l-1}|| / ||S_{l-1}|| between consecutive
  block outputs S, the stream sum for HC (what the readout and final norm see). S_0 is the
  embedding (times rate for HC, the sum of its expanded copies).
- norm: median ||S_l||.
Each is stored as [5th, 50th, 95th] percentile over all tokens of all images. Plus linear CKA
between the 12 normed block outputs (the readout's per-layer features) on a token subsample.

Robustness (a shared direction or one huge channel can push raw cosines and CKA toward 1):
- cos_sub_c, cos_block_c: the same cosines after subtracting each scan's mean token.
- mean_share: energy of the scan's mean token / mean token energy (per scan; 1 = identical tokens).
- top1_share, top4_share: share of a token's energy in its largest 1 and 4 channels (1/384 if flat).
- top_ratio: largest |channel| / median |channel| per token (massive activations are ~100-1000x).
- top_channel: the channel most often largest, and how often.
- cka_c: CKA on readout features centered per scan, so differences between scans' means drop out.

usage: python scripts/layer_similarity.py --runs a17/batched-nt l1/dhc4-w384-matched --seeds 7338 1
writes <output>/<run>/s<seed>/similarity.json
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from nanobrain.data import BrainNpzDataset, process_sample
from nanobrain.eval.run import load_model
from nanobrain.model import patchify3d

ROOT = Path("/data/nanobrain/FOMO300K_images")
OUT = Path("/data/dojo/nanobrain/output")
Q = torch.tensor([0.05, 0.5, 0.95])


def quantiles(values: list[list[torch.Tensor]]) -> list[list[float]]:
    return [torch.quantile(torch.cat(v).float(), Q.to(v[0].device)).tolist() for v in values]


def cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double() - x.double().mean(0)
    y = y.double() - y.double().mean(0)
    return ((y.T @ x).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm())).item()


def tokens(model, image, mask):
    """Patch tokens of one image, normalized as in ReadoutViTMAE3D.readout."""
    m = mask > 0
    values = image[m]
    values = values / values.max()
    values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)
    img = torch.zeros_like(image)
    img[m] = values
    patches = patchify3d(img[None], model.patch_size)
    in_mask = patchify3d(m[None], model.patch_size).any(dim=2)[0]
    return patches[:, in_mask], model.coord_grid[:, in_mask]


@torch.inference_mode()
def measure(model, images, tokens_per_image: int) -> dict:
    depth = len(model.blocks)
    k = model.class_tokens
    rate = model.hc_rate or 1
    cap = {"in": [], "out": [], "blk": []}
    hooks = []
    for block in model.blocks:
        for norm, branch in [(block.norm1, block.attn), (block.norm2, block.mlp)]:
            hooks.append(norm.register_forward_pre_hook(lambda m, a: cap["in"].append(a[0].float())))
            hooks.append(branch.register_forward_hook(lambda m, a, o: cap["out"].append(o.float())))
        hooks.append(block.register_forward_hook(lambda m, a, o: cap["blk"].append(o.float())))

    cos_sub = [[] for _ in range(2 * depth - 1)]
    update = [[] for _ in range(2 * depth)]
    cos_block = [[] for _ in range(depth)]
    delta = [[] for _ in range(depth)]
    norm = [[] for _ in range(depth + 1)]
    feats = [[] for _ in range(depth)]
    # robustness: centered cosines, the shared (mean-token) direction, single dominant channels
    cos_sub_c = [[] for _ in range(2 * depth - 1)]
    cos_block_c = [[] for _ in range(depth)]
    mean_share = [[] for _ in range(depth + 1)]
    top1_share = [[] for _ in range(depth + 1)]
    top4_share = [[] for _ in range(depth + 1)]
    top_ratio = [[] for _ in range(depth + 1)]
    argmax = [[] for _ in range(depth + 1)]
    feats_c = [[] for _ in range(depth)]
    check_err = 0.0
    gen = torch.Generator(device="cuda").manual_seed(0)
    for image, mask in images:
        patches, coord = tokens(model, image, mask)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = model.forward_encoder(patches, coord)  # also fires the hooks, so clear after
        for v in cap.values():
            v.clear()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = model.patch_embed(patches)
            x = model.pos_embed(x, coord)
            if k:
                x = torch.cat([model.cls_token.expand(1, -1, -1), x], dim=1)
            emb = x.float()
            if model.hc_rate:
                x = x[:, :, None, :].expand(-1, -1, model.hc_rate, -1)
            for block in model.blocks:
                x = block(x)
        ins = [t[0, k:] for t in cap["in"]]
        outs = [t[0, k:] for t in cap["out"]]
        sums = [rate * emb[0, k:]] + [(t.sum(dim=2) if model.hc_rate else t)[0, k:] for t in cap["blk"]]
        assert len(ins) == len(outs) == 2 * depth and len(sums) == depth + 1

        # checks: the hooks see the real forward
        dtype = model.norm.weight.dtype
        with torch.autocast("cuda", dtype=torch.bfloat16):
            last = model.norm(sums[-1].to(dtype))
        err = (last.float() - ref[0, k:].float()).abs().max().item()
        check_err = max(check_err, err)
        assert err < 2e-2, err
        if not model.hc_rate:
            # plain pre-norm: each sublayer input is the previous input plus the previous branch output
            for j in range(1, 2 * depth):
                err = (ins[j] - ins[j - 1] - outs[j - 1]).norm() / ins[j].norm()
                assert err < 2e-2, (j, err)

        for j in range(1, 2 * depth):
            cos_sub[j - 1].append(F.cosine_similarity(ins[j], ins[j - 1], dim=-1))
        for j in range(2 * depth):
            update[j].append(outs[j].norm(dim=-1) / ins[j].norm(dim=-1).clamp_min(1e-6))
        for j in range(1, 2 * depth):
            a, b = ins[j] - ins[j].mean(0), ins[j - 1] - ins[j - 1].mean(0)
            cos_sub_c[j - 1].append(F.cosine_similarity(a, b, dim=-1))
        for layer in range(depth + 1):
            s = sums[layer]
            norm[layer].append(s.norm(dim=-1))
            sq = s.square()
            # energy of this scan's mean token over the mean token energy: 1 = all tokens identical
            mean_share[layer].append((s.mean(0).square().sum() / sq.sum(-1).mean()).view(1))
            top = sq.topk(4, dim=-1).values
            top1_share[layer].append(top[:, 0] / sq.sum(-1))
            top4_share[layer].append(top.sum(-1) / sq.sum(-1))
            top_ratio[layer].append(s.abs().amax(-1) / s.abs().median(-1).values.clamp_min(1e-6))
            argmax[layer].append(s.abs().argmax(-1))
        idx = torch.randperm(sums[0].shape[0], generator=gen, device="cuda")[:tokens_per_image]
        for layer in range(1, depth + 1):
            cos_block[layer - 1].append(F.cosine_similarity(sums[layer], sums[layer - 1], dim=-1))
            delta[layer - 1].append((sums[layer] - sums[layer - 1]).norm(dim=-1) / sums[layer - 1].norm(dim=-1).clamp_min(1e-6))
            a, b = sums[layer] - sums[layer].mean(0), sums[layer - 1] - sums[layer - 1].mean(0)
            cos_block_c[layer - 1].append(F.cosine_similarity(a, b, dim=-1))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model.norm(sums[layer].to(dtype)).float()
            feats[layer - 1].append(f[idx])
            feats_c[layer - 1].append((f - f.mean(0))[idx])

    for h in hooks:
        h.remove()
    feats = [torch.cat(f) for f in feats]
    feats_c = [torch.cat(f) for f in feats_c]
    top_channel = []
    for v in argmax:
        counts = torch.bincount(torch.cat(v))
        top_channel.append([counts.argmax().item(), (counts.max() / counts.sum()).item()])
    return {
        "check_err": check_err,
        "cos_sub_c": quantiles(cos_sub_c),
        "cos_block_c": quantiles(cos_block_c),
        "mean_share": quantiles(mean_share),
        "top1_share": quantiles(top1_share),
        "top4_share": quantiles(top4_share),
        "top_ratio": quantiles(top_ratio),
        "top_channel": top_channel,
        "cka_c": [[cka(a, b) for b in feats_c] for a in feats_c],
        "n_images": len(images),
        "n_tokens": sum(t.numel() for t in cos_block[0]),
        "n_cka_tokens": feats[0].shape[0],
        "cos_sub": quantiles(cos_sub),
        "update": quantiles(update),
        "cos_block": quantiles(cos_block),
        "delta": quantiles(delta),
        "norm": [torch.cat(v).median().item() for v in norm],
        "cka": [[cka(a, b) for b in feats] for a in feats],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7338, 1, 2, 3])
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--tokens-per-image", type=int, default=256)
    args = parser.parse_args()

    dataset = BrainNpzDataset(ROOT, ROOT / "filelist_subsampled.txt")
    index = random.Random(0).sample(range(len(dataset)), args.num_images)
    images, paths = [], []
    for i in index:
        sample = dataset[i]
        out = process_sample(sample)
        images.append((out["image"].cuda(), out["mask"].cuda()))
        paths.append(sample["path"])
    print("loaded", len(images), "images", flush=True)

    for run in args.runs:
        for seed in args.seeds:
            run_dir = OUT / run / f"s{seed}"
            model = load_model(run_dir / "checkpoint-last.pth").cuda().eval()
            result = measure(model, images, args.tokens_per_image)
            result.update(run=run, seed=seed, hc_rate=model.hc_rate, paths=paths)
            (run_dir / "similarity.json").write_text(json.dumps(result))
            cs = [round(q[1], 3) for q in result["cos_sub"]]
            print(run, seed, "median cos_sub", cs, flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
