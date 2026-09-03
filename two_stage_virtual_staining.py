from __future__ import annotations

import json
import itertools
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from cyclegan_core import (
    ImagePool,
    PatchDiscriminator,
    RGBMemoryCache,
    ResNetGenerator,
    denormalize,
    init_weights,
    masked_l1,
    paired_spatial_augment,
    set_requires_grad,
    simple_ssim,
    slide_key,
)


def collect_matched_pairs(data_dir, image_ext="png"):
    data_dir = Path(data_dir)
    unstain = sorted((data_dir / "unstain").glob(f"*.{image_ext}"))
    hne_by_name = {p.name: p for p in (data_dir / "hne").glob(f"*.{image_ext}")}
    pairs = [(p, hne_by_name[p.name]) for p in unstain if p.name in hne_by_name]
    if not pairs:
        raise RuntimeError(f"No filename-matched pairs under {data_dir}")
    return pairs


def split_pairs_by_slide(pairs, val_fraction=0.1, seed=42):
    groups = defaultdict(list)
    for pair in pairs:
        groups[slide_key(pair[0])].append(pair)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        raise RuntimeError("Slide-wise split requires at least two slides.")
    n_val = min(len(keys) - 1, max(1, round(len(keys) * val_fraction)))
    val_keys = set(keys[:n_val])
    train = [pair for key in keys if key not in val_keys for pair in groups[key]]
    val = [pair for key in keys if key in val_keys for pair in groups[key]]
    return train, val


def prepare_physical_view(image, params):
    output_size = int(params["input_size"])
    original_size = int(params["original_size"])
    source_mpp = float(params["source_mpp"])
    target_mpp = float(params["target_mpp"])
    view_size = round(output_size * target_mpp / source_mpp)
    if view_size > original_size:
        raise ValueError(
            f"{output_size}px at {target_mpp} MPP requires {view_size}px at "
            f"{source_mpp} MPP, larger than original_size={original_size}."
        )
    if view_size < original_size:
        offset = (original_size - view_size) // 2
        image = TF.crop(image, offset, offset, view_size, view_size)
    if image.size != (output_size, output_size):
        image = TF.resize(
            image,
            [output_size, output_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    return image


def estimate_global_od_max(cache, paths, params, label):
    count = min(len(paths), int(params["od_calibration_images"]))
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    max_od = -math.log(1 / 255)
    bins = 4096
    histogram = np.zeros(bins, dtype=np.int64)
    edges = np.linspace(0, max_od, bins + 1)
    threshold = float(params["od_background_threshold"])
    for index in tqdm(indices, desc=f"Calibrating {label} OD at target MPP"):
        image = prepare_physical_view(cache.get(paths[int(index)]), params)
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        tissue = gray < threshold
        if tissue.any():
            od = -np.log(np.clip(gray[tissue], 1 / 255, 1.0))
            histogram += np.histogram(od, bins=edges)[0]
    total = histogram.sum()
    if total == 0:
        raise RuntimeError(f"No tissue pixels found while calibrating {label} OD.")
    target = float(params["od_quantile"]) * total
    bin_index = min(int(np.searchsorted(np.cumsum(histogram), target)), bins - 1)
    return max(float(edges[bin_index + 1]), 0.05)


def image_to_od(image, od_max):
    gray = TF.rgb_to_grayscale(TF.to_tensor(image), num_output_channels=1)
    od = -torch.log(gray.clamp(1 / 255, 1.0))
    od = (od / float(od_max)).clamp(0, 1)
    return od * 2 - 1


class TwoStagePairDataset(Dataset):
    def __init__(self, pairs, cache, params, unstain_od_max, hne_od_max, training, mode):
        self.pairs = list(pairs)
        self.cache = cache
        self.params = params
        self.unstain_od_max = float(unstain_od_max)
        self.hne_od_max = float(hne_od_max)
        self.training = bool(training)
        self.mode = mode
        if mode not in {"structure", "color"}:
            raise ValueError("mode must be 'structure' or 'color'")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        unstain_path, hne_path = self.pairs[index]
        unstain = prepare_physical_view(self.cache.get(unstain_path), self.params)
        hne = prepare_physical_view(self.cache.get(hne_path), self.params)
        if self.training:
            unstain, hne = paired_spatial_augment(unstain, hne)
        unstain_od = image_to_od(unstain, self.unstain_od_max)
        if self.mode == "structure":
            hne_od = image_to_od(hne, self.hne_od_max)
            return unstain_od, hne_od
        hne_rgb = TF.to_tensor(hne) * 2 - 1
        return unstain_od, hne_rgb


def build_two_stage_dataloaders(params):
    pairs = collect_matched_pairs(params["data_dir"], params["image_ext"])
    if len(pairs) > int(params["image_max_count"]):
        pairs = random.Random(params["seed"]).sample(pairs, int(params["image_max_count"]))
    train_pairs, val_pairs = split_pairs_by_slide(pairs, params["val_fraction"], params["seed"])
    all_paths = [p for pair in train_pairs + val_pairs for p in pair]
    cache = RGBMemoryCache(
        all_paths,
        params["original_size"],
        params["preload_images"],
        params["max_cache_gib"],
    )
    unstain_od_max = estimate_global_od_max(
        cache, [p[0] for p in train_pairs], params, "Unstain"
    )
    hne_od_max = estimate_global_od_max(cache, [p[1] for p in train_pairs], params, "H&E")

    def loader(dataset, shuffle, drop_last):
        return DataLoader(
            dataset,
            batch_size=params["batch_size"],
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
        )

    datasets = {
        "structure_train": TwoStagePairDataset(
            train_pairs, cache, params, unstain_od_max, hne_od_max, True, "structure"
        ),
        "structure_val": TwoStagePairDataset(
            val_pairs, cache, params, unstain_od_max, hne_od_max, False, "structure"
        ),
        "color_train": TwoStagePairDataset(
            train_pairs, cache, params, unstain_od_max, hne_od_max, True, "color"
        ),
        "color_val": TwoStagePairDataset(
            val_pairs, cache, params, unstain_od_max, hne_od_max, False, "color"
        ),
    }
    result = {
        "structure_train": loader(datasets["structure_train"], True, True),
        "structure_val": loader(datasets["structure_val"], False, False),
        "color_train": loader(datasets["color_train"], True, True),
        "color_val": loader(datasets["color_val"], False, False),
        "unstain_od_max": unstain_od_max,
        "hne_od_max": hne_od_max,
        "cache": cache,
    }
    view_size = round(params["input_size"] * params["target_mpp"] / params["source_mpp"])
    print(
        f"pairs: train={len(train_pairs):,}, val={len(val_pairs):,}; "
        f"view={view_size}px@{params['source_mpp']}MPP -> "
        f"{params['input_size']}px@{params['target_mpp']}MPP; "
        f"OD_MAX unstain={unstain_od_max:.4f}, H&E={hne_od_max:.4f}"
    )
    return result


def od_gradient_loss(x, y):
    x_dx, y_dx = x[:, :, :, 1:] - x[:, :, :, :-1], y[:, :, :, 1:] - y[:, :, :, :-1]
    x_dy, y_dy = x[:, :, 1:, :] - x[:, :, :-1, :], y[:, :, 1:, :] - y[:, :, :-1, :]
    return F.l1_loss(x_dx, y_dx) + F.l1_loss(x_dy, y_dy)


class StructureCycleGANTrainer:
    """True 1-channel paired CycleGAN for Unstain OD ↔ H&E OD."""

    def __init__(self, params, train_loader, val_loader, od_max, device):
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.od_max = float(od_max)
        self.device = device
        self.output_dir = Path(params["output_dir"])
        self.checkpoint_dir = Path(params["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.G_AB = ResNetGenerator(1, params["ngf"], params["residual_blocks"]).to(device)
        self.G_BA = ResNetGenerator(1, params["ngf"], params["residual_blocks"]).to(device)
        self.D_A = PatchDiscriminator(1, params["ndf"]).to(device)
        self.D_B = PatchDiscriminator(1, params["ndf"]).to(device)
        for model in (self.G_AB, self.G_BA, self.D_A, self.D_B):
            model.apply(init_weights)

        self.opt_g = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=params["lr_g"], betas=(params["beta1"], params["beta2"]),
        )
        self.opt_d_a = torch.optim.Adam(
            self.D_A.parameters(), lr=params["lr_d"], betas=(params["beta1"], params["beta2"])
        )
        self.opt_d_b = torch.optim.Adam(
            self.D_B.parameters(), lr=params["lr_d"], betas=(params["beta1"], params["beta2"])
        )

        def lr_rule(epoch):
            if epoch < params["decay_start_epoch"]:
                return 1.0
            span = max(1, params["num_epochs"] - params["decay_start_epoch"])
            return max(0.0, 1.0 - (epoch - params["decay_start_epoch"] + 1) / span)

        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(opt, lr_rule)
            for opt in (self.opt_g, self.opt_d_a, self.opt_d_b)
        ]
        self.gan_loss = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.pool_a = ImagePool(params["pool_size"])
        self.pool_b = ImagePool(params["pool_size"])
        self.amp_enabled = device.type == "cuda"
        self.scaler_g = torch.amp.GradScaler(device=device.type, enabled=self.amp_enabled)
        self.scaler_d_a = torch.amp.GradScaler(device=device.type, enabled=self.amp_enabled)
        self.scaler_d_b = torch.amp.GradScaler(device=device.type, enabled=self.amp_enabled)
        self.start_epoch = 0
        self.history = []
        self.best_score = math.inf
        print(
            f"1-channel structure G_AB={sum(p.numel() for p in self.G_AB.parameters()) / 1e6:.2f}M, "
            f"D_B={sum(p.numel() for p in self.D_B.parameters()) / 1e6:.2f}M"
        )

    def background_losses(self, real_a, real_b, fake_a, fake_b):
        real_a_01, real_b_01 = denormalize(real_a), denormalize(real_b)
        mask_a = (real_a_01 < self.params["a_background_od_threshold"]).to(real_a.dtype)
        mask_b = (real_b_01 < self.params["b_background_od_threshold"]).to(real_b.dtype)
        kernel = int(self.params["background_mask_blur_kernel"])
        if kernel > 1:
            mask_a = F.avg_pool2d(mask_a, kernel, 1, kernel // 2)
            mask_b = F.avg_pool2d(mask_b, kernel, 1, kernel // 2)
        loss_ab = masked_l1(denormalize(fake_b), 0.0, mask_a)
        loss_ba = masked_l1(denormalize(fake_a), 0.0, mask_b)
        return loss_ab, loss_ba

    def generator_step(self, real_a, real_b):
        set_requires_grad([self.D_A, self.D_B], False)
        self.opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
            fake_b = self.G_AB(real_a)
            fake_a = self.G_BA(real_b)
            rec_a = self.G_BA(fake_b)
            rec_b = self.G_AB(fake_a)
            idt_a = self.G_BA(real_a)
            idt_b = self.G_AB(real_b)

            pred_b, pred_a = self.D_B(fake_b), self.D_A(fake_a)
            gan_ab = self.gan_loss(pred_b, torch.ones_like(pred_b))
            gan_ba = self.gan_loss(pred_a, torch.ones_like(pred_a))
            cycle_a, cycle_b = self.l1(rec_a, real_a), self.l1(rec_b, real_b)
            identity_a, identity_b = self.l1(idt_a, real_a), self.l1(idt_b, real_b)
            paired_ab, paired_ba = self.l1(fake_b, real_b), self.l1(fake_a, real_a)
            ssim_ab, ssim_ba = simple_ssim(fake_b, real_b), simple_ssim(fake_a, real_a)
            gradient_ab = od_gradient_loss(fake_b, real_b)
            gradient_ba = od_gradient_loss(fake_a, real_a)
            background_ab, background_ba = self.background_losses(real_a, real_b, fake_a, fake_b)

            loss_g = (
                self.params["lambda_gan"] * (gan_ab + gan_ba)
                + self.params["lambda_cycle"] * (cycle_a + cycle_b)
                + self.params["lambda_identity"] * (identity_a + identity_b)
                + self.params["lambda_paired"] * (paired_ab + paired_ba)
                + self.params["lambda_ssim"] * ((1 - ssim_ab) + (1 - ssim_ba))
                + self.params["lambda_gradient"] * (gradient_ab + gradient_ba)
                + self.params["lambda_background"] * (background_ab + background_ba)
            )
        self.scaler_g.scale(loss_g).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()
        metrics = {
            "G": loss_g, "gan_AB": gan_ab, "gan_BA": gan_ba,
            "cycle_A": cycle_a, "cycle_B": cycle_b,
            "identity_A": identity_a, "identity_B": identity_b,
            "paired_AB": paired_ab, "paired_BA": paired_ba,
            "ssim_AB": ssim_ab, "ssim_BA": ssim_ba,
            "gradient_AB": gradient_ab, "gradient_BA": gradient_ba,
            "background_AB": background_ab, "background_BA": background_ba,
        }
        return {key: value.detach() for key, value in metrics.items()}, fake_a.detach(), fake_b.detach()

    def discriminator_step(self, discriminator, optimizer, scaler, real, fake):
        set_requires_grad(discriminator, True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
            pred_real, pred_fake = discriminator(real), discriminator(fake)
            loss = 0.5 * (
                self.gan_loss(pred_real, torch.ones_like(pred_real))
                + self.gan_loss(pred_fake, torch.zeros_like(pred_fake))
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    def train_epoch(self, epoch):
        for model in (self.G_AB, self.G_BA, self.D_A, self.D_B):
            model.train()
        totals = defaultdict(float)
        pbar = tqdm(self.train_loader, desc=f"Structure epoch {epoch + 1}/{self.params['num_epochs']}")
        for step, (real_a, real_b) in enumerate(pbar, start=1):
            real_a = real_a.to(self.device, non_blocking=True)
            real_b = real_b.to(self.device, non_blocking=True)
            metrics, fake_a, fake_b = self.generator_step(real_a, real_b)
            metrics["D_A"] = self.discriminator_step(
                self.D_A, self.opt_d_a, self.scaler_d_a, real_a, self.pool_a.query(fake_a)
            )
            metrics["D_B"] = self.discriminator_step(
                self.D_B, self.opt_d_b, self.scaler_d_b, real_b, self.pool_b.query(fake_b)
            )
            for key, value in metrics.items():
                totals[key] += float(value)
            pbar.set_postfix(
                G=f"{totals['G'] / step:.3f}", D=f"{(totals['D_A'] + totals['D_B']) / (2 * step):.3f}",
                pair=f"{totals['paired_AB'] / step:.3f}", SSIM=f"{totals['ssim_AB'] / step:.3f}",
            )
        return {key: value / len(self.train_loader) for key, value in totals.items()}

    @torch.no_grad()
    def validate(self):
        self.G_AB.eval(); self.G_BA.eval()
        totals = defaultdict(float)
        count = 0
        preview = None
        for real_a, real_b in self.val_loader:
            real_a = real_a.to(self.device, non_blocking=True)
            real_b = real_b.to(self.device, non_blocking=True)
            fake_b = self.G_AB(real_a)
            fake_a = self.G_BA(real_b)
            rec_a = self.G_BA(fake_b)
            rec_b = self.G_AB(fake_a)
            background_ab, background_ba = self.background_losses(real_a, real_b, fake_a, fake_b)
            metrics = {
                "paired_AB": self.l1(fake_b, real_b),
                "paired_BA": self.l1(fake_a, real_a),
                "ssim_AB": simple_ssim(fake_b, real_b),
                "ssim_BA": simple_ssim(fake_a, real_a),
                "gradient_AB": od_gradient_loss(fake_b, real_b),
                "gradient_BA": od_gradient_loss(fake_a, real_a),
                "cycle_A": self.l1(rec_a, real_a),
                "cycle_B": self.l1(rec_b, real_b),
                "background_AB": background_ab,
                "background_BA": background_ba,
            }
            batch = real_a.shape[0]
            for key, value in metrics.items():
                totals[key] += float(value) * batch
            count += batch
            if preview is None:
                preview = tuple(
                    x.detach().float().cpu()
                    for x in (real_a, fake_b, real_b, rec_a, real_b, fake_a, real_a, rec_b)
                )
        result = {key: value / count for key, value in totals.items()}
        result["selection_score"] = (
            result["paired_AB"] + (1 - result["ssim_AB"])
            + 0.5 * result["gradient_AB"] + result["background_AB"]
        )
        return result, preview

    def save_preview(self, epoch, preview):
        labels = [
            "Unstain OD", "Fake H&E OD", "Real H&E OD", "Recovered Unstain OD",
            "Real H&E OD", "Fake Unstain OD", "Real Unstain OD", "Recovered H&E OD",
        ]
        rows = min(self.params["preview_count"], preview[0].shape[0])
        fig, axes = plt.subplots(rows, 8, figsize=(24, 3 * rows), squeeze=False)
        for row in range(rows):
            for col, batch in enumerate(preview):
                axes[row, col].imshow(denormalize(batch[row, 0]).numpy(), cmap="gray", vmin=0, vmax=1)
                axes[row, col].set_title(labels[col])
                axes[row, col].axis("off")
        fig.suptitle(f"1-channel structure CycleGAN — epoch {epoch + 1}", y=1.01)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"epoch_{epoch + 1:04d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_checkpoint(self, epoch, is_best):
        state = {
            "epoch": epoch, "G_AB": self.G_AB.state_dict(), "G_BA": self.G_BA.state_dict(),
            "D_A": self.D_A.state_dict(), "D_B": self.D_B.state_dict(),
            "opt_g": self.opt_g.state_dict(), "opt_d_a": self.opt_d_a.state_dict(),
            "opt_d_b": self.opt_d_b.state_dict(),
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
            "history": self.history, "best_score": self.best_score,
            "od_max": self.od_max, "params": self.params,
        }
        torch.save(state, self.checkpoint_dir / "latest.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")
        if (epoch + 1) % self.params["save_every"] == 0:
            torch.save(state, self.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt")

    def fit(self):
        for epoch in range(self.start_epoch, self.params["num_epochs"]):
            train_metrics = self.train_epoch(epoch)
            val_metrics, preview = self.validate()
            for scheduler in self.schedulers:
                scheduler.step()
            row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
            self.history.append(row)
            is_best = val_metrics["selection_score"] < self.best_score
            if is_best:
                self.best_score = val_metrics["selection_score"]
            self.save_checkpoint(epoch, is_best)
            self.save_preview(epoch, preview)
            print("train:", {k: round(v, 4) for k, v in train_metrics.items()})
            print("val:", {k: round(v, 4) for k, v in val_metrics.items()})
            if is_best:
                print(f"new best forward H&E OD: epoch {epoch + 1}")
            with (self.output_dir / "history.json").open("w") as file:
                json.dump(self.history, file, indent=2)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, 3, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.down(x)


class DetailRefinementBlock(nn.Module):
    """High-resolution residual refinement without normalization-induced smoothing."""

    def __init__(self, channels, residual_scale=0.2):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.detail = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
        )

    def forward(self, x):
        return x + self.residual_scale * self.detail(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, refine=False):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        self.refine = DetailRefinementBlock(out_channels) if refine else nn.Identity()

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(self.conv(torch.cat([x, skip], dim=1)))


class ODColorizer(nn.Module):
    def __init__(self, base=32, input_channels=1, detail_refinement=False):
        super().__init__()
        self.input_channels = int(input_channels)
        self.detail_refinement = bool(detail_refinement)
        self.e1 = ConvBlock(self.input_channels, base)
        self.e2 = DownBlock(base, base * 2)
        self.e3 = DownBlock(base * 2, base * 4)
        self.e4 = DownBlock(base * 4, base * 8)
        self.bottleneck = nn.Sequential(
            DownBlock(base * 8, base * 8),
            ConvBlock(base * 8, base * 8),
        )
        self.u4 = UpBlock(base * 8, base * 8, base * 8, self.detail_refinement)
        self.u3 = UpBlock(base * 8, base * 4, base * 4, self.detail_refinement)
        self.u2 = UpBlock(base * 4, base * 2, base * 2, self.detail_refinement)
        self.u1 = UpBlock(base * 2, base, base, self.detail_refinement)
        self.output_refine = (
            DetailRefinementBlock(base) if self.detail_refinement else nn.Identity()
        )
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(base, 3, 7), nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self.bottleneck(e4)
        x = self.u1(self.u2(self.u3(self.u4(b, e4), e3), e2), e1)
        return self.output(self.output_refine(x))


def rgb_gradient_loss(fake, real):
    return od_gradient_loss(fake, real)


def rgb_laplacian_loss(fake, real):
    kernel = fake.new_tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(fake.shape[1], 1, 1, 1)
    fake_laplacian = F.conv2d(
        F.pad(fake, (1, 1, 1, 1), mode="reflect"), kernel, groups=fake.shape[1]
    )
    real_laplacian = F.conv2d(
        F.pad(real, (1, 1, 1, 1), mode="reflect"), kernel, groups=real.shape[1]
    )
    return F.l1_loss(fake_laplacian, real_laplacian)


class ColorizerTrainer:
    def __init__(self, params, train_loader, val_loader, structure_generator, device):
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.structure_generator = structure_generator.eval()
        for parameter in self.structure_generator.parameters():
            parameter.requires_grad = False
        self.device = device
        self.output_dir = Path(params["output_dir"])
        self.checkpoint_dir = Path(params["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.input_channels = int(params.get("input_channels", 1))
        if self.input_channels != 1:
            raise ValueError("Colorizer input must be predicted H&E OD only (1 channel).")
        self.G = ODColorizer(
            params["base_channels"], self.input_channels,
            detail_refinement=params.get("detail_refinement", True),
        ).to(device)
        self.D = PatchDiscriminator(self.input_channels + 3, params["ndf"]).to(device)
        self.G.apply(init_weights)
        self.D.apply(init_weights)
        self.opt_g = torch.optim.Adam(
            self.G.parameters(), lr=params["lr_g"], betas=(params["beta1"], params["beta2"])
        )
        self.opt_d = torch.optim.Adam(
            self.D.parameters(), lr=params["lr_d"], betas=(params["beta1"], params["beta2"])
        )
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_g, params["num_epochs"]
        )
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_d, params["num_epochs"]
        )
        self.gan_loss = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.amp_enabled = device.type == "cuda"
        self.scaler_g = torch.amp.GradScaler(device=device.type, enabled=self.amp_enabled)
        self.scaler_d = torch.amp.GradScaler(device=device.type, enabled=self.amp_enabled)
        self.start_epoch = 0
        self.history = []
        self.best_score = math.inf

    def background_loss(self, hne_od, fake_rgb):
        od_01 = denormalize(hne_od).mean(dim=1, keepdim=True)
        mask = (od_01 < self.params["background_od_threshold"]).to(fake_rgb.dtype)
        kernel = int(self.params["background_mask_blur_kernel"])
        if kernel > 1:
            mask = F.avg_pool2d(mask, kernel, 1, kernel // 2)
        return masked_l1(denormalize(fake_rgb), 1.0, mask)

    def train_epoch(self, epoch):
        self.G.train()
        self.D.train()
        self.structure_generator.eval()
        totals = defaultdict(float)
        pbar = tqdm(self.train_loader, desc=f"Color epoch {epoch + 1}/{self.params['num_epochs']}")
        for step, (unstain_od, real_rgb) in enumerate(pbar, start=1):
            unstain_od = unstain_od.to(self.device, non_blocking=True)
            real_rgb = real_rgb.to(self.device, non_blocking=True)
            with torch.no_grad():
                predicted_od = self.structure_generator(unstain_od)
            condition = predicted_od

            for parameter in self.D.parameters():
                parameter.requires_grad = False
            self.opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                fake_rgb = self.G(condition)
                pred_fake = self.D(torch.cat([condition, fake_rgb], dim=1))
                gan = self.gan_loss(pred_fake, torch.ones_like(pred_fake))
                rgb_l1 = self.l1(fake_rgb, real_rgb)
                ssim_loss = 1 - simple_ssim(fake_rgb, real_rgb)
                gradient = rgb_gradient_loss(fake_rgb, real_rgb)
                laplacian = rgb_laplacian_loss(fake_rgb, real_rgb)
                background = self.background_loss(predicted_od, fake_rgb)
                loss_g = (
                    self.params["lambda_gan"] * gan
                    + self.params["lambda_rgb"] * rgb_l1
                    + self.params["lambda_ssim"] * ssim_loss
                    + self.params["lambda_gradient"] * gradient
                    + self.params["lambda_laplacian"] * laplacian
                    + self.params["lambda_background"] * background
                )
            self.scaler_g.scale(loss_g).backward()
            self.scaler_g.step(self.opt_g)
            self.scaler_g.update()

            for parameter in self.D.parameters():
                parameter.requires_grad = True
            self.opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                pred_real = self.D(torch.cat([condition, real_rgb], dim=1))
                pred_fake = self.D(torch.cat([condition, fake_rgb.detach()], dim=1))
                loss_d = 0.5 * (
                    self.gan_loss(pred_real, torch.ones_like(pred_real))
                    + self.gan_loss(pred_fake, torch.zeros_like(pred_fake))
                )
            self.scaler_d.scale(loss_d).backward()
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()

            metrics = {
                "G": loss_g, "D": loss_d, "gan": gan, "rgb_l1": rgb_l1,
                "ssim_loss": ssim_loss, "ssim_score": 1 - ssim_loss,
                "gradient": gradient, "laplacian": laplacian,
                "background": background,
            }
            for key, value in metrics.items():
                totals[key] += float(value.detach())
            pbar.set_postfix(
                G=f"{totals['G'] / step:.3f}", D=f"{totals['D'] / step:.3f}",
                SSIM=f"{totals['ssim_score'] / step:.3f}",
            )
        return {key: value / len(self.train_loader) for key, value in totals.items()}

    @torch.no_grad()
    def validate(self):
        self.G.eval()
        self.structure_generator.eval()
        totals = defaultdict(float)
        count = 0
        preview = None
        for unstain_od, real_rgb in self.val_loader:
            unstain_od = unstain_od.to(self.device, non_blocking=True)
            real_rgb = real_rgb.to(self.device, non_blocking=True)
            predicted_od = self.structure_generator(unstain_od)
            fake_predicted = self.G(predicted_od)
            pred_l1 = self.l1(fake_predicted, real_rgb)
            pred_ssim = simple_ssim(fake_predicted, real_rgb)
            gradient = rgb_gradient_loss(fake_predicted, real_rgb)
            laplacian = rgb_laplacian_loss(fake_predicted, real_rgb)
            background = self.background_loss(predicted_od, fake_predicted)
            batch = unstain_od.shape[0]
            for key, value in {
                "predicted_rgb_l1": pred_l1,
                "predicted_ssim": pred_ssim,
                "gradient": gradient,
                "laplacian": laplacian,
                "background": background,
            }.items():
                totals[key] += float(value) * batch
            count += batch
            if preview is None:
                preview = tuple(
                    x.detach().float().cpu()
                    for x in (unstain_od, predicted_od, fake_predicted, real_rgb)
                )
        metrics = {key: value / count for key, value in totals.items()}
        metrics["selection_score"] = (
            metrics["predicted_rgb_l1"] + (1 - metrics["predicted_ssim"])
            + 0.5 * metrics["gradient"] + 0.25 * metrics["laplacian"]
            + self.params["lambda_background"] * metrics["background"] / max(self.params["lambda_rgb"], 1)
        )
        return metrics, preview

    def save_preview(self, epoch, preview):
        labels = [
            "Unstain OD (Stage 1 only)", "Predicted H&E OD",
            "Final RGB from predicted OD", "Real RGB H&E",
        ]
        rows = min(self.params["preview_count"], preview[0].shape[0])
        columns = len(preview)
        fig, axes = plt.subplots(rows, columns, figsize=(3.5 * columns, 3.2 * rows), squeeze=False)
        for row in range(rows):
            for col, batch in enumerate(preview):
                if batch.shape[1] == 1:
                    image = denormalize(batch[row, 0]).numpy().clip(0, 1)
                    axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=1)
                else:
                    image = denormalize(batch[row]).permute(1, 2, 0).numpy().clip(0, 1)
                    axes[row, col].imshow(image)
                axes[row, col].set_title(labels[col])
                axes[row, col].axis("off")
        fig.suptitle(f"Two-stage virtual H&E — color epoch {epoch + 1}", y=1.01)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"epoch_{epoch + 1:04d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_checkpoint(self, epoch, is_best):
        state = {
            "epoch": epoch,
            "G_color": self.G.state_dict(),
            "D_color": self.D.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
            "scaler_g": self.scaler_g.state_dict(),
            "scaler_d": self.scaler_d.state_dict(),
            "history": self.history,
            "best_score": self.best_score,
            "params": self.params,
        }
        torch.save(state, self.checkpoint_dir / "latest.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")
        if (epoch + 1) % self.params["save_every"] == 0:
            torch.save(state, self.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt")

    def fit(self):
        for epoch in range(self.start_epoch, self.params["num_epochs"]):
            train_metrics = self.train_epoch(epoch)
            val_metrics, preview = self.validate()
            self.scheduler_g.step()
            self.scheduler_d.step()
            row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
            self.history.append(row)
            is_best = val_metrics["selection_score"] < self.best_score
            if is_best:
                self.best_score = val_metrics["selection_score"]
            self.save_checkpoint(epoch, is_best)
            self.save_preview(epoch, preview)
            print("train:", {k: round(v, 4) for k, v in train_metrics.items()})
            print("val:", {k: round(v, 4) for k, v in val_metrics.items()})
            if is_best:
                print(f"new best colorizer: epoch {epoch + 1}")
            with (self.output_dir / "history.json").open("w") as file:
                json.dump(self.history, file, indent=2)


def build_structure_trainer(params, data, device):
    return StructureCycleGANTrainer(
        params,
        data["structure_train"],
        data["structure_val"],
        data["unstain_od_max"],
        device,
    )
