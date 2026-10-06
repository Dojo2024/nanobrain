import argparse
import collections
import datetime
import itertools
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Sampler, default_collate
from torch.utils.flop_counter import FlopCounterMode

import nanobrain.utils.misc as misc
from nanobrain.data import BrainNpzDataset, process_sample
from nanobrain.model import ViTMAE3D, make_mask_groups, patches_to_volume
from nanobrain.visualization import plot_mask_pred

logger = logging.getLogger()

DEFAULT_CONFIG = Path(__file__).parent / "config/default_pretrain.yaml"


def main(args: DictConfig):
    # setup
    assert int(os.environ.get("WORLD_SIZE", 1)) == 1, "distributed training not supported"
    device = torch.device(args.device)
    misc.random_seed(args.seed)

    if args.name and not args.output_dir.endswith(args.name):
        args.output_dir = f"{args.output_dir}/{args.name}"
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    out_cfg_path = output_dir / "config.yaml"
    if out_cfg_path.exists():
        prev_cfg = OmegaConf.load(out_cfg_path)
        assert args == prev_cfg, "current config doesn't match previous config"
    else:
        OmegaConf.save(args, out_cfg_path)

    if args.wandb:
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.name,
            notes=args.notes,
            config=OmegaConf.to_container(args),
        )

    misc.setup_logging(logger, output_dir / "log.txt")

    logger.info("pretraining nanobrain")
    logger.info(f"start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"cwd: {Path.cwd()}")
    logger.info(f"sha: {misc.git_sha()}")
    logger.info(f"config:\n{OmegaConf.to_yaml(args)}")

    # data loader
    dataset = BrainNpzDataset(args.data_root, args.train_filelist)
    logger.info(f"train dataset: {len(dataset)} images")
    if args.modality_power is not None:
        sampler = ModalitySampler(dataset.paths, args.modality_power)
    else:
        sampler = None
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=list,  # no collate, variable shapes. lambda can't be pickled to workers
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # model
    model = ViTMAE3D.from_config(args)
    model.to(device)
    logger.info(f"model:\n{model}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"num params: {num_params / 1e6:.1f}M")

    batch_flops = count_batch_flops(args, model, device)
    logger.info(f"flops per batch (fwd + bwd): {batch_flops / 1e12:.2f} TFLOP")

    if args.compile:
        # in place compile keeps the state dict keys
        model.compile()

    # optimizer
    # batch size counts sequences, following mae_st which scales by repeated samples
    total_batch_size = args.batch_size * args.num_samples * args.accum_iter
    logger.info(
        f"total batch size: {total_batch_size} = "
        f"{args.batch_size} images x {args.num_samples} samples x {args.accum_iter} accum"
    )

    if not args.get("lr"):
        args.lr = args.base_lr * total_batch_size / 256
        logger.info(f"lr: {args.lr:.2e} = {args.base_lr:.2e} x {total_batch_size} / 256")
    else:
        logger.info(f"lr: {args.lr:.2e}")

    assert args.optimizer == "muon" or args.muon_lr_mult == 1.0, "muon_lr_mult needs optimizer muon"
    assert args.optimizer == "muon" or args.muon_impl == "torch", "muon_impl needs optimizer muon"
    param_groups = misc.get_param_groups(
        model, muon=args.optimizer == "muon", muon_lr_mult=args.muon_lr_mult
    )
    misc.update_lr(param_groups, args.lr)
    misc.update_wd(param_groups, args.weight_decay)
    # cast or else it corrupts the checkpoint
    betas = tuple(args.betas) if args.betas is not None else None
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, betas=betas, fused=args.fused_adamw)
    else:
        assert args.optimizer == "muon", args.optimizer
        optimizer = misc.MuonAdamW(
            param_groups, betas=betas, fused=args.fused_adamw, impl=args.muon_impl
        )
        num_muon = sum(p.numel() for g in optimizer.muon.param_groups for p in g["params"])
        num_adamw = sum(p.numel() for g in optimizer.adamw.param_groups for p in g["params"])
        logger.info(f"optimizer: muon {num_muon / 1e6:.2f}M params, adamw {num_adamw / 1e6:.2f}M")

    epoch_num_batches = len(train_loader)
    steps_per_epoch = math.ceil(epoch_num_batches / args.accum_iter)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    schedule_class = {"cosine": misc.WarmupThenCosine, "linear": misc.WarmupThenLinear}
    lr_schedule = schedule_class[args.lr_decay](
        base_value=args.lr,
        final_value=args.min_lr,
        total_iters=total_steps,
        warmup_iters=warmup_steps,
    )
    logger.info(f"full schedule: epochs = {args.epochs} (steps = {total_steps})")
    logger.info(f"warmup: epochs = {args.warmup_epochs} (steps = {warmup_steps})")
    logger.info(f"lr decay: {args.lr_decay}")
    if args.time_budget:
        logger.info(
            f"time budget: {args.time_budget} s spans the {total_steps} step schedule, "
            "training stops at the budget"
        )

    # loss scaling not needed for bfloat16 (according to timm)
    if args.amp and args.amp_dtype != "bfloat16":
        loss_scaler = torch.GradScaler(device.type)
    else:
        loss_scaler = None
    assert loss_scaler is None or args.optimizer == "adamw", "muon has no GradScaler support"

    # load checkpoint/resume training
    misc.load_model(args, model, optimizer, loss_scaler)
    # elapsed time restarts on resume, which would replay the schedule
    assert not (args.time_budget and args.start_epoch), "a time budget run can't resume"
    assert not (args.time_budget and args.stop_epoch), "a time budget run stops at the budget"
    # the EMA copies are not saved for resuming
    assert not (args.ema_decays and args.start_epoch), "an ema run can't resume"
    ema = misc.ModelEMA(model, args.ema_decays) if args.ema_decays else None

    logger.info(f"start training for {args.epochs} epochs")
    start_time = time.monotonic()
    # with a time budget, epochs run until the budget runs out
    if args.time_budget:
        epochs = itertools.count(args.start_epoch)
    else:
        assert not args.stop_epoch or args.stop_epoch <= args.epochs, "stop_epoch is past epochs"
        epochs = range(args.start_epoch, args.stop_epoch or args.epochs)
    for epoch in epochs:
        train_stats = train_one_epoch(
            args,
            model,
            train_loader,
            optimizer,
            loss_scaler,
            lr_schedule,
            epoch,
            device,
            batch_flops,
            start_time,
            total_steps,
            ema,
        )

        merged_stats = {"epoch": epoch, **train_stats}
        with (output_dir / "log.json").open("a") as f:
            print(json.dumps(merged_stats), file=f)

        misc.save_model(args, epoch, model, optimizer, loss_scaler)
        if ema is not None:
            misc.save_ema(args, epoch, model, ema)
        if args.time_budget and time.monotonic() - start_time >= args.time_budget:
            break

    total_time = time.monotonic() - start_time
    logger.info(f"done! training time: {datetime.timedelta(seconds=int(total_time))}")


def train_one_epoch(
    args: DictConfig,
    model: nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_scaler: torch.GradScaler | None,
    lr_schedule: Sequence[float],
    epoch: int,
    device: torch.device,
    batch_flops: float,
    start_time: float,
    total_steps: int,
    ema: misc.ModelEMA | None = None,
):
    model.train()
    if args.mask_groups:
        patch_grid_size = [size // args.patch_size for size in args.grid_size]
        mask_groups = make_mask_groups(patch_grid_size, args.mask_groups).to(device)
    else:
        mask_groups = None

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("grad", misc.SmoothedValue())
    metric_logger.add_meter("loss", misc.SmoothedValue())
    metric_logger.add_meter(
        "vol/s", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    metric_logger.add_meter(
        "tflop/s", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    metric_logger.add_meter(
        "watts", misc.SmoothedValue(window_size=1, fmt="{value:.0f} ({global_avg:.0f})")
    )
    header = f"Train: [{epoch}]"

    throughput = misc.ThroughputMeter(
        {"vol": args.batch_size, "tflop": batch_flops / 1e12}, device=device
    )

    epoch_num_batches = len(data_loader)
    steps_per_epoch = math.ceil(epoch_num_batches / args.accum_iter)

    print_freq = args.get("print_freq", 100) if not args.debug else 1
    num_batches = epoch_num_batches if not args.debug else 10
    amp_dtype = getattr(torch, args.amp_dtype)
    use_cuda = device.type == "cuda"

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header, total_steps=num_batches)
    ):
        batch_step = batch_idx + 1
        log_step = batch_step % print_freq == 0 or batch_step == num_batches
        update_in_epoch = batch_idx // args.accum_iter
        group_size = min(args.accum_iter, num_batches - update_in_epoch * args.accum_iter)
        need_update = batch_step % args.accum_iter == 0 or batch_step == num_batches
        global_step = epoch * steps_per_epoch + update_in_epoch
        if args.time_budget:
            # key the schedule to elapsed wall time, so it finishes decaying at the budget
            elapsed = time.monotonic() - start_time
            schedule_step = min(int(elapsed / args.time_budget * total_steps), total_steps)
        else:
            schedule_step = global_step
        lr = lr_schedule[schedule_step]
        if need_update:
            misc.update_lr(optimizer.param_groups, lr)

        if use_cuda:
            batch = misc.send_data(batch, device)

        # run volume processing on gpu, not cpu workers
        batch = default_collate([process_sample(sample) for sample in batch])
        images = batch["image"]
        masks = batch["mask"]

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss = model(
                images,
                masks,
                num_visible=args.num_visible,
                num_predict=args.num_predict,
                num_samples=args.num_samples,
                with_state=False,
                mask_groups=mask_groups,
                norm_target=args.norm_target,
                stats_weight=args.stats_weight,
            )

        loss_for_log = loss.detach()
        torch._assert_async(torch.isfinite(loss_for_log), "non-finite loss")

        grad_norm = misc.backward_step(
            loss / group_size,
            optimizer,
            scaler=loss_scaler,
            need_update=need_update,
            max_norm=args.clip_grad,
        )

        if need_update and ema is not None:
            ema.update()

        # log tensors without syncing, values are only read on log steps
        metric_logger.update(loss=loss_for_log)
        if need_update:
            metric_logger.update(lr=lr, grad=grad_norm)

        throughput.step()
        if log_step:
            metric_logger.update(**throughput.compute())

        if need_update and log_step and args.wandb:
            wandb.log(
                {
                    "train/loss": metric_logger.loss.value,
                    "train/lr": lr,
                    "train/grad": metric_logger.grad.value,
                    "train/vol_per_sec": metric_logger.meters["vol/s"].value,
                    "train/tflop_per_sec": metric_logger.meters["tflop/s"].value,
                    "train/watts": metric_logger.meters["watts"].value,
                },
                step=int(1000 * (epoch + batch_step / epoch_num_batches)),
            )

        # stop after an optimizer step, so the saved checkpoint has no pending gradients
        if need_update and args.time_budget and time.monotonic() - start_time >= args.time_budget:
            logger.info(f"time budget reached at step {batch_step}/{num_batches} of epoch {epoch}")
            break

    # plot a few images from the last batch
    num_examples = args.plot_nrow * args.plot_ncol
    is_plot_epoch = args.plot_period and (epoch % args.plot_period == 0 or epoch == args.epochs - 1)
    if num_examples and is_plot_epoch:
        model.eval()
        example_images = images[:num_examples]
        example_masks = masks[:num_examples]
        with (
            torch.no_grad(),
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp),
        ):
            _, state = model(
                example_images,
                example_masks,
                num_visible=args.num_visible,
                num_predict=args.plot_num_predict,
                num_samples=1,
                mask_groups=mask_groups,
                norm_target=args.norm_target,
                stats_weight=args.stats_weight,
            )
        model.train()
        plot_paths = make_plots(args, example_images, example_masks, state, epoch)
        if args.wandb:
            wandb.log(
                {f"train/{name}": wandb.Image(str(path)) for name, path in plot_paths.items()},
                step=1000 * (epoch + 1),
            )

    logger.info(f"Averaged stats: {metric_logger}")
    return {f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}


class ModalitySampler(Sampler):
    """Each epoch draws len(paths) images: scan type m gets a quota proportional to its image
    count ** power, filled by cycling through shuffled orders of that type's images, so repeats
    are as even as possible. power 1 gives every image once per epoch, like shuffle; power 0 is
    uniform over types. The scan type is the last _ token of the file name, e.g.
    sub-01_ses-01_T1w.npz -> T1w."""

    def __init__(self, paths: Sequence[str], power: float):
        self.indices = collections.defaultdict(list)
        for index, path in enumerate(paths):
            self.indices[Path(path).stem.rsplit("_", 1)[-1]].append(index)
        total = sum(len(indices) ** power for indices in self.indices.values())
        self.quotas = {
            modality: round(len(paths) * len(indices) ** power / total)
            for modality, indices in self.indices.items()
        }
        # one epoch stays len(paths) samples, so the lr schedule is unchanged
        largest = max(self.quotas, key=self.quotas.get)
        self.quotas[largest] += len(paths) - sum(self.quotas.values())
        self.num_samples = len(paths)
        for modality, quota in sorted(self.quotas.items(), key=lambda item: -item[1]):
            count = len(self.indices[modality])
            logger.info(
                f"modality {modality}: {count} images ({count / len(paths):.1%}), "
                f"{quota} samples per epoch ({quota / len(paths):.1%}, {quota / count:.2f}x each)"
            )

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        samples = []
        for modality, quota in self.quotas.items():
            indices = torch.tensor(self.indices[modality])
            repeats = math.ceil(quota / len(indices))
            orders = [indices[torch.randperm(len(indices))] for _ in range(repeats)]
            samples.append(torch.cat(orders)[:quota])
        samples = torch.cat(samples)
        yield from samples[torch.randperm(len(samples))].tolist()


def count_batch_flops(args: DictConfig, model: nn.Module, device: torch.device) -> float:
    # fwd + bwd flops of one training batch, on dummy images where every patch is in the mask
    amp_dtype = getattr(torch, args.amp_dtype)
    images = torch.zeros(args.batch_size, *args.grid_size, device=device)
    masks = torch.ones(args.batch_size, *args.grid_size, device=device)
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp):
            loss = model(
                images,
                masks,
                num_visible=args.num_visible,
                num_predict=args.num_predict,
                num_samples=args.num_samples,
                with_state=False,
                norm_target=args.norm_target,
                stats_weight=args.stats_weight,
            )
        loss.backward()
    model.zero_grad(set_to_none=True)
    return flop_counter.get_total_flops()


def make_plots(
    args: DictConfig,
    images: Tensor,
    masks: Tensor,
    state: dict[str, Tensor],
    epoch: int,
) -> dict[str, Path]:
    grid_size = tuple(args.grid_size)
    patch_size = args.patch_size
    vis_ids = state["vis_ids"]
    target_ids = state["target_ids"]
    pred = patches_to_volume(state["pred_patches"].float(), target_ids, grid_size, patch_size)
    visible_mask = patches_to_volume(
        torch.ones_like(state["vis_patches"]), vis_ids, grid_size, patch_size
    )
    pred_mask = patches_to_volume(
        torch.ones_like(state["target_patches"]), target_ids, grid_size, patch_size
    )

    images = images.float().cpu().numpy()
    masks = masks.cpu().numpy()
    pred = pred.cpu().numpy()
    visible_mask = visible_mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()

    plot_dir = Path(args.output_dir) / "plots"
    plot_paths = {}
    for view in ["sag", "cor", "ax"]:
        fig = plot_mask_pred(
            images,
            pred,
            visible_mask,
            pred_mask=pred_mask,
            img_mask=masks,
            view=view,
            nrow=args.plot_nrow,
            ncol=args.plot_ncol,
        )
        path = plot_dir / f"mae_{view}_{epoch:05d}.png"
        fig.savefig(path)
        plt.close(fig)
        plot_paths[f"mae_{view}"] = path
    return plot_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default=None)
    parser.add_argument("--overrides", type=str, default=None, nargs="+")
    args = parser.parse_args()
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    if args.cfg_path:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.load(args.cfg_path))
    if args.overrides:
        cfg = OmegaConf.unsafe_merge(cfg, OmegaConf.from_dotlist(args.overrides))
    main(cfg)
