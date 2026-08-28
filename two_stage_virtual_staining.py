from __future__ import annotations

import json
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
    CycleGANTrainer,
    PatchDiscriminator,
    RGBMemoryCache,
    denormalize,
    init_weights,
    masked_l1,
    paired_spatial_augment,
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


def image_to_od3(image, od_max):
    gray = TF.rgb_to_grayscale(TF.to_tensor(image), num_output_channels=1)
    od = -torch.log(gray.clamp(1 / 255, 1.0))
    od = (od / float(od_max)).clamp(0, 1)
    return (od * 2 - 1).repeat(3, 1, 1)


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
        unstain_od = image_to_od3(unstain, self.unstain_od_max)
        hne_od = image_to_od3(hne, self.hne_od_max)
        if self.mode == "structure":
            return unstain_od, hne_od
        hne_rgb = TF.to_tensor(hne) * 2 - 1
        return unstain_od, hne_od, hne_rgb


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


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ODColorizer(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.e1 = ConvBlock(3, base)
        self.e2 = DownBlock(base, base * 2)
        self.e3 = DownBlock(base * 2, base * 4)
        self.e4 = DownBlock(base * 4, base * 8)
        self.bottleneck = nn.Sequential(
            DownBlock(base * 8, base * 8),
            ConvBlock(base * 8, base * 8),
        )
        self.u4 = UpBlock(base * 8, base * 8, base * 8)
        self.u3 = UpBlock(base * 8, base * 4, base * 4)
        self.u2 = UpBlock(base * 4, base * 2, base * 2)
        self.u1 = UpBlock(base * 2, base, base)
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(base, 3, 7), nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b = self.bottleneck(e4)
        return self.output(self.u1(self.u2(self.u3(self.u4(b, e4), e3), e2), e1))


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

        self.G = ODColorizer(params["base_channels"]).to(device)
        self.D = PatchDiscriminator(6, params["ndf"]).to(device)
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

    def predicted_probability(self, epoch):
        start = int(self.params["predicted_od_start_epoch"])
        ramp = max(1, int(self.params["predicted_od_ramp_epochs"]))
        if epoch < start:
            return 0.0
        return self.params["max_predicted_od_probability"] * min(1.0, (epoch - start + 1) / ramp)

    @torch.no_grad()
    def mix_od_inputs(self, unstain_od, real_hne_od, probability):
        if probability <= 0:
            return real_hne_od, real_hne_od
        predicted = self.structure_generator(unstain_od)
        mask = (torch.rand(unstain_od.shape[0], 1, 1, 1, device=self.device) < probability)
        return torch.where(mask, predicted, real_hne_od), predicted

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
        probability = self.predicted_probability(epoch)
        totals = defaultdict(float)
        pbar = tqdm(self.train_loader, desc=f"Color epoch {epoch + 1}/{self.params['num_epochs']}")
        for step, (unstain_od, real_hne_od, real_rgb) in enumerate(pbar, start=1):
            unstain_od = unstain_od.to(self.device, non_blocking=True)
            real_hne_od = real_hne_od.to(self.device, non_blocking=True)
            real_rgb = real_rgb.to(self.device, non_blocking=True)
            input_od, _ = self.mix_od_inputs(unstain_od, real_hne_od, probability)

            for parameter in self.D.parameters():
                parameter.requires_grad = False
            self.opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                fake_rgb = self.G(input_od)
                pred_fake = self.D(torch.cat([input_od, fake_rgb], dim=1))
                gan = self.gan_loss(pred_fake, torch.ones_like(pred_fake))
                rgb_l1 = self.l1(fake_rgb, real_rgb)
                fake_mid = F.interpolate(denormalize(fake_rgb), (256, 256), mode="area")
                real_mid = F.interpolate(denormalize(real_rgb), (256, 256), mode="area")
                ssim_loss = 1 - simple_ssim(fake_mid * 2 - 1, real_mid * 2 - 1)
                background = self.background_loss(input_od, fake_rgb)
                loss_g = (
                    self.params["lambda_gan"] * gan
                    + self.params["lambda_rgb"] * rgb_l1
                    + self.params["lambda_ssim"] * ssim_loss
                    + self.params["lambda_background"] * background
                )
            self.scaler_g.scale(loss_g).backward()
            self.scaler_g.step(self.opt_g)
            self.scaler_g.update()

            for parameter in self.D.parameters():
                parameter.requires_grad = True
            self.opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                pred_real = self.D(torch.cat([input_od, real_rgb], dim=1))
                pred_fake = self.D(torch.cat([input_od, fake_rgb.detach()], dim=1))
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
                "background": background,
            }
            for key, value in metrics.items():
                totals[key] += float(value.detach())
            pbar.set_postfix(
                G=f"{totals['G'] / step:.3f}", D=f"{totals['D'] / step:.3f}",
                SSIM=f"{totals['ssim_score'] / step:.3f}", pred=f"{probability:.2f}",
            )
        result = {key: value / len(self.train_loader) for key, value in totals.items()}
        result["predicted_od_probability"] = probability
        return result

    @torch.no_grad()
    def validate(self):
        self.G.eval()
        self.structure_generator.eval()
        totals = defaultdict(float)
        count = 0
        preview = None
        for unstain_od, real_hne_od, real_rgb in self.val_loader:
            unstain_od = unstain_od.to(self.device, non_blocking=True)
            real_hne_od = real_hne_od.to(self.device, non_blocking=True)
            real_rgb = real_rgb.to(self.device, non_blocking=True)
            predicted_od = self.structure_generator(unstain_od)
            fake_predicted = self.G(predicted_od)
            fake_oracle = self.G(real_hne_od)
            pred_l1 = self.l1(fake_predicted, real_rgb)
            oracle_l1 = self.l1(fake_oracle, real_rgb)
            pred_ssim = simple_ssim(fake_predicted, real_rgb)
            oracle_ssim = simple_ssim(fake_oracle, real_rgb)
            background = self.background_loss(predicted_od, fake_predicted)
            batch = unstain_od.shape[0]
            for key, value in {
                "predicted_rgb_l1": pred_l1,
                "oracle_rgb_l1": oracle_l1,
                "predicted_ssim": pred_ssim,
                "oracle_ssim": oracle_ssim,
                "background": background,
            }.items():
                totals[key] += float(value) * batch
            count += batch
            if preview is None:
                preview = tuple(
                    x.detach().float().cpu()
                    for x in (unstain_od, predicted_od, real_hne_od, fake_predicted, fake_oracle, real_rgb)
                )
        metrics = {key: value / count for key, value in totals.items()}
        metrics["selection_score"] = (
            metrics["predicted_rgb_l1"] + (1 - metrics["predicted_ssim"])
            + self.params["lambda_background"] * metrics["background"] / max(self.params["lambda_rgb"], 1)
        )
        return metrics, preview

    def save_preview(self, epoch, preview):
        labels = [
            "Unstain OD", "Predicted H&E OD", "Real H&E OD",
            "Final RGB from predicted OD", "RGB from real OD", "Real RGB H&E",
        ]
        rows = min(self.params["preview_count"], preview[0].shape[0])
        fig, axes = plt.subplots(rows, 6, figsize=(19, 3.2 * rows), squeeze=False)
        for row in range(rows):
            for col, batch in enumerate(preview):
                image = denormalize(batch[row]).permute(1, 2, 0).numpy().clip(0, 1)
                axes[row, col].imshow(image, cmap="gray" if col < 3 else None)
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
    return CycleGANTrainer(
        params,
        data["structure_train"],
        data["structure_val"],
        data["unstain_od_max"],
        device,
    )

