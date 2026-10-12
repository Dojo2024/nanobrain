"""Check data.augment_intensity: off path bit-identical to HEAD, on path finite and z-scored, timing."""
import sys, time
import torch
sys.path.insert(0, sys.argv[1])
import data_head
from nanobrain.data import BrainNpzDataset, process_sample

root = "/data/nanobrain/FOMO300K_images"
ds = BrainNpzDataset(root, f"{root}/filelist_subsampled.txt")
idx = [i * (len(ds) // 32) for i in range(32)]
samples = []
for i in idx:
    s = ds[i]
    s = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in s.items()}
    samples.append(s)

same = all(torch.equal(process_sample(s)["image"], data_head.process_sample(s)["image"])
           and torch.equal(process_sample(s)["mask"], data_head.process_sample(s)["mask"]) for s in samples)
print("off path bit-identical to HEAD on 32 images:", same)

torch.manual_seed(0)
corrs, means, stds, finite = [], [], [], True
for s in samples:
    a = process_sample(s, intensity_aug=True)
    o = process_sample(s)
    assert torch.equal(a["mask"], o["mask"])
    m = a["mask"] > 0
    finite &= bool(torch.isfinite(a["image"]).all())
    x, y = a["image"][m], o["image"][m]
    means.append(x.mean().item()); stds.append(x.std().item())
    corrs.append(torch.corrcoef(torch.stack([x, y]))[0, 1].item())
    assert (a["image"][~m] == 0).all()
print("aug finite:", finite, "| outside mask all zero: True")
print(f"in-mask mean range [{min(means):+.3f}, {max(means):+.3f}], std range [{min(stds):.3f}, {max(stds):.3f}]")
print(f"corr(aug, original) in mask: min {min(corrs):.3f} median {sorted(corrs)[16]:.3f} max {max(corrs):.3f}")

def bench(flag, reps=5):
    for _ in range(2):
        [process_sample(s, intensity_aug=flag) for s in samples]
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps):
        [process_sample(s, intensity_aug=flag) for s in samples]
    torch.cuda.synchronize(); return (time.perf_counter() - t) / reps * 1000
off, on = bench(False), bench(True)
print(f"32 images per step: off {off:.1f} ms, on {on:.1f} ms, extra {on - off:.1f} ms ({(on - off) / 32:.2f} ms/image)")
