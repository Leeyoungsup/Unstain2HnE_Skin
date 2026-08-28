from __future__ import annotations

import itertools
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slide_key(path: str | Path) -> str:
    return re.sub(r"_\d{5}_x-?\d+_y-?\d+$", "", Path(path).stem)


def collect_domain_paths(data_dir: str | Path, image_ext: str = "png"):
    data_dir = Path(data_dir)
    unstain = sorted((data_dir / "unstain").glob(f"*.{image_ext}"))
    hne = sorted((data_dir / "hne").glob(f"*.{image_ext}"))
    if not unstain or not hne:
        raise RuntimeError(f"No images found under {data_dir}/unstain and {data_dir}/hne")
    return unstain, hne


def split_domains_by_slide(unstain_paths, hne_paths, val_fraction=0.1, seed=42):
    all_keys = sorted({slide_key(p) for p in unstain_paths + hne_paths})
    random.Random(seed).shuffle(all_keys)
    if len(all_keys) < 2:
        raise RuntimeError("Slide-wise split requires at least two slides.")
    n_val = min(len(all_keys) - 1, max(1, round(len(all_keys) * val_fraction)))
    val_keys = set(all_keys[:n_val])
    train_a = [p for p in unstain_paths if slide_key(p) not in val_keys]
    train_b = [p for p in hne_paths if slide_key(p) not in val_keys]
    val_a = [p for p in unstain_paths if slide_key(p) in val_keys]
    val_b = [p for p in hne_paths if slide_key(p) in val_keys]
    return train_a, train_b, val_a, val_b


class RGBMemoryCache:
    """Decode source patches once and share them across train/validation datasets."""

    def __init__(self, paths, original_size, preload=True, max_cache_gib=64):
        self.original_size = int(original_size)
        self.images = {}
        self.preload = bool(preload)
        unique_paths = sorted({Path(p) for p in paths})
        estimated_gib = len(unique_paths) * self.original_size**2 * 3 / 1024**3
        if self.preload and estimated_gib > max_cache_gib:
            raise MemoryError(
                f"Decoded cache requires about {estimated_gib:.1f} GiB, "
                f"larger than max_cache_gib={max_cache_gib}."
            )
        if self.preload:
            for path in tqdm(unique_paths, desc="Preloading RGB originals into RAM"):
                self.images[path] = self._read(path)
            print(f"RAM cache: {estimated_gib:.2f} GiB estimated decoded RGB")

    def _read(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
        if image.size != (self.original_size, self.original_size):
            image = TF.resize(
                image,
                [self.original_size, self.original_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
        return image

    def get(self, path):
        path = Path(path)
        image = self.images[path] if self.preload else self._read(path)
        return image.copy()


def estimate_od_max(cache, unstain_paths, calibration_images=256, quantile=0.995, threshold=0.98):
    count = min(len(unstain_paths), calibration_images)
    indices = np.linspace(0, len(unstain_paths) - 1, count, dtype=int)
    values = []
    for index in tqdm(indices, desc="Estimating fixed Unstain OD range"):
        image = cache.get(unstain_paths[int(index)]).resize((128, 128), Image.Resampling.BILINEAR)
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        tissue = gray < threshold
        if tissue.any():
            values.append(-np.log(np.clip(gray[tissue], 1 / 255, 1.0)))
    if not values:
        raise RuntimeError("OD calibration found no tissue pixels.")
    return max(float(np.quantile(np.concatenate(values), quantile)), 0.05)


def tissue_fraction(image, domain, check_size=64, white_threshold=0.98):
    small = image.resize((check_size, check_size), Image.Resampling.BILINEAR)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    if domain == "A":
        gray = rgb.mean(axis=2)
        return float(np.mean(gray < white_threshold))
    brightness = rgb.mean(axis=2)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    return float(np.mean((brightness < 0.92) | (saturation > 0.06)))


def choose_tissue_crop(
    image,
    crop_size,
    domain,
    training,
    retry_count,
    min_fraction,
    sampling_mode="tissue",
    background_max_fraction=0.02,
    boundary_min_fraction=0.02,
    boundary_max_fraction=0.50,
):
    width, height = image.size
    if width < crop_size or height < crop_size:
        raise ValueError(f"Image {image.size} is smaller than crop_size={crop_size}")
    max_left, max_top = width - crop_size, height - crop_size
    if training:
        candidates = [
            (random.randint(0, max_top), random.randint(0, max_left))
            for _ in range(retry_count)
        ]
    else:
        candidates = [
            (max_top // 2, max_left // 2),
            (0, 0),
            (0, max_left),
            (max_top, 0),
            (max_top, max_left),
        ]
    best = candidates[0]
    best_score = -math.inf
    boundary_target = (boundary_min_fraction + boundary_max_fraction) / 2
    for top, left in candidates:
        crop = TF.crop(image, top, left, crop_size, crop_size)
        fraction = tissue_fraction(crop, domain)

        if sampling_mode == "background":
            score = -fraction
            accepted = fraction <= background_max_fraction
        elif sampling_mode == "boundary":
            score = -abs(fraction - boundary_target)
            accepted = boundary_min_fraction <= fraction <= boundary_max_fraction
        else:
            score = fraction
            accepted = fraction >= min_fraction

        if score > best_score:
            best, best_score = (top, left), score
        if accepted:
            return top, left
    return best


def spatial_augment(image):
    angle = random.choice((0, 90, 180, 270))
    if angle:
        image = TF.rotate(image, angle)
    if random.random() < 0.5:
        image = TF.hflip(image)
    if random.random() < 0.5:
        image = TF.vflip(image)
    return image


def paired_spatial_augment(image_a, image_b):
    angle = random.choice((0, 90, 180, 270))
    if angle:
        image_a = TF.rotate(image_a, angle)
        image_b = TF.rotate(image_b, angle)
    if random.random() < 0.5:
        image_a, image_b = TF.hflip(image_a), TF.hflip(image_b)
    if random.random() < 0.5:
        image_a, image_b = TF.vflip(image_a), TF.vflip(image_b)
    return image_a, image_b


def unstain_to_od3(image, od_max):
    rgb = TF.to_tensor(image)
    gray = TF.rgb_to_grayscale(rgb, num_output_channels=1)
    od = -torch.log(gray.clamp(1 / 255, 1.0))
    od = (od / od_max).clamp(0, 1)
    # Fixed global scaling: low-OD background -> -1, dense tissue -> +1.
    return (od * 2 - 1).repeat(3, 1, 1)


def hne_to_tensor(image):
    return TF.to_tensor(image) * 2 - 1


class UnpairedCycleDataset(Dataset):
    """Samples A and B independently; filename registration is not used for training."""

    def __init__(self, a_paths, b_paths, cache, params, od_max):
        self.a_paths = list(a_paths)
        self.b_paths = list(b_paths)
        self.cache = cache
        self.output_size = int(params["input_size"])
        self.original_size = int(params["original_size"])
        self.source_mpp = float(params.get("source_mpp", 0.5))
        self.target_mpp = float(params.get("target_mpp", self.source_mpp))
        self.view_size = round(self.output_size * self.target_mpp / self.source_mpp)
        self.od_max = float(od_max)
        self.retry_count = int(params["crop_retry_count"])
        self.min_fraction = float(params["min_crop_tissue_fraction"])
        self.background_probability = float(params.get("background_crop_probability", 0.10))
        self.boundary_probability = float(params.get("boundary_crop_probability", 0.20))
        self.background_max_fraction = float(params.get("background_max_tissue_fraction", 0.02))
        self.boundary_min_fraction = float(params.get("boundary_min_tissue_fraction", 0.02))
        self.boundary_max_fraction = float(params.get("boundary_max_tissue_fraction", 0.50))

        if self.background_probability + self.boundary_probability > 1:
            raise ValueError("background + boundary crop probabilities must be <= 1")
        if self.view_size > self.original_size:
            raise ValueError(
                f"A {self.output_size}px output at {self.target_mpp} MPP requires a "
                f"{self.view_size}px source view, larger than original_size={self.original_size}."
            )

    def __len__(self):
        return max(len(self.a_paths), len(self.b_paths))

    def _view(self, path, domain):
        image = self.cache.get(path)
        if self.view_size < self.original_size:
            draw = random.random()
            if draw < self.background_probability:
                sampling_mode = "background"
            elif draw < self.background_probability + self.boundary_probability:
                sampling_mode = "boundary"
            else:
                sampling_mode = "tissue"
            top, left = choose_tissue_crop(
                image,
                self.view_size,
                domain,
                True,
                self.retry_count,
                self.min_fraction,
                sampling_mode=sampling_mode,
                background_max_fraction=self.background_max_fraction,
                boundary_min_fraction=self.boundary_min_fraction,
                boundary_max_fraction=self.boundary_max_fraction,
            )
            image = TF.crop(image, top, left, self.view_size, self.view_size)
        if image.size != (self.output_size, self.output_size):
            image = TF.resize(
                image,
                [self.output_size, self.output_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        return spatial_augment(image)

    def __getitem__(self, index):
        # Domain B is deliberately independent of A, including crop and augmentation.
        a_path = self.a_paths[index % len(self.a_paths)]
        b_index = random.randrange(len(self.b_paths))
        b_path = self.b_paths[b_index]
        if len(self.b_paths) > 1 and b_path.name == a_path.name:
            b_index = (b_index + random.randrange(1, len(self.b_paths))) % len(self.b_paths)
            b_path = self.b_paths[b_index]
        a = unstain_to_od3(self._view(a_path, "A"), self.od_max)
        b = hne_to_tensor(self._view(b_path, "B"))
        return a, b


class PairedCycleDataset(Dataset):
    """Filename-matched domains with the same field of view and augmentation."""

    def __init__(self, a_paths, b_paths, cache, params, od_max):
        b_by_name = {p.name: p for p in b_paths}
        self.pairs = [(p, b_by_name[p.name]) for p in a_paths if p.name in b_by_name]
        if not self.pairs:
            raise RuntimeError("No filename-matched training pairs were found.")
        self.cache = cache
        self.output_size = int(params["input_size"])
        self.original_size = int(params["original_size"])
        self.source_mpp = float(params.get("source_mpp", 0.5))
        self.target_mpp = float(params.get("target_mpp", self.source_mpp))
        self.view_size = round(self.output_size * self.target_mpp / self.source_mpp)
        self.od_max = float(od_max)
        self.retry_count = int(params["crop_retry_count"])
        self.min_fraction = float(params["min_crop_tissue_fraction"])
        if self.view_size > self.original_size:
            raise ValueError(
                f"A {self.output_size}px output at {self.target_mpp} MPP requires a "
                f"{self.view_size}px source view, larger than original_size={self.original_size}."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a_path, b_path = self.pairs[index]
        image_a, image_b = self.cache.get(a_path), self.cache.get(b_path)
        if self.view_size < self.original_size:
            top, left = choose_tissue_crop(
                image_a,
                self.view_size,
                "A",
                True,
                self.retry_count,
                self.min_fraction,
            )
            image_a = TF.crop(image_a, top, left, self.view_size, self.view_size)
            image_b = TF.crop(image_b, top, left, self.view_size, self.view_size)
        if image_a.size != (self.output_size, self.output_size):
            image_a = TF.resize(
                image_a,
                [self.output_size, self.output_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            image_b = TF.resize(
                image_b,
                [self.output_size, self.output_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        image_a, image_b = paired_spatial_augment(image_a, image_b)
        return unstain_to_od3(image_a, self.od_max), hne_to_tensor(image_b)


class PairedValidationDataset(Dataset):
    """Fixed corresponding crops are only for diagnostics; they never enter train loss."""

    def __init__(self, a_paths, b_paths, cache, params, od_max):
        b_by_name = {p.name: p for p in b_paths}
        self.pairs = [(p, b_by_name[p.name]) for p in a_paths if p.name in b_by_name]
        if not self.pairs:
            raise RuntimeError("No filename-matched validation pairs were found.")
        self.cache = cache
        self.output_size = int(params["input_size"])
        self.original_size = int(params["original_size"])
        self.source_mpp = float(params.get("source_mpp", 0.5))
        self.target_mpp = float(params.get("target_mpp", self.source_mpp))
        self.view_size = round(self.output_size * self.target_mpp / self.source_mpp)
        self.od_max = float(od_max)
        self.retry_count = int(params["crop_retry_count"])
        self.min_fraction = float(params["min_crop_tissue_fraction"])
        if self.view_size > self.original_size:
            raise ValueError(
                f"A {self.output_size}px output at {self.target_mpp} MPP requires a "
                f"{self.view_size}px source view, larger than original_size={self.original_size}."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a_path, b_path = self.pairs[index]
        a_image, b_image = self.cache.get(a_path), self.cache.get(b_path)
        if self.view_size < self.original_size:
            top, left = choose_tissue_crop(
                a_image,
                self.view_size,
                "A",
                False,
                self.retry_count,
                self.min_fraction,
            )
            a_image = TF.crop(a_image, top, left, self.view_size, self.view_size)
            b_image = TF.crop(b_image, top, left, self.view_size, self.view_size)
        if a_image.size != (self.output_size, self.output_size):
            a_image = TF.resize(
                a_image,
                [self.output_size, self.output_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            b_image = TF.resize(
                b_image,
                [self.output_size, self.output_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        return unstain_to_od3(a_image, self.od_max), hne_to_tensor(b_image)


def build_dataloaders(params):
    unstain, hne = collect_domain_paths(params["data_dir"], params["image_ext"])
    max_count = int(params["image_max_count"])
    if len(unstain) > max_count:
        unstain = random.Random(params["seed"]).sample(unstain, max_count)
    if len(hne) > max_count:
        hne = random.Random(params["seed"] + 1).sample(hne, max_count)
    train_a, train_b, val_a, val_b = split_domains_by_slide(
        unstain, hne, params["val_fraction"], params["seed"]
    )
    cache = RGBMemoryCache(
        train_a + train_b + val_a + val_b,
        params["original_size"],
        params["preload_images"],
        params["max_cache_gib"],
    )
    od_max = estimate_od_max(
        cache,
        train_a,
        params["od_calibration_images"],
        params["od_quantile"],
        params["od_background_threshold"],
    )
    if params.get("paired_training", False):
        train_set = PairedCycleDataset(train_a, train_b, cache, params, od_max)
        training_mode = "paired"
    else:
        train_set = UnpairedCycleDataset(train_a, train_b, cache, params, od_max)
        training_mode = "unpaired"
    val_set = PairedValidationDataset(val_a, val_b, cache, params, od_max)
    train_loader = DataLoader(
        train_set,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(
        f"train A={len(train_a):,}, train B={len(train_b):,}, "
        f"val pairs={len(val_set):,}, OD_MAX={od_max:.4f}, "
        f"mode={training_mode}, view={train_set.view_size}px@{train_set.source_mpp}MPP "
        f"-> {train_set.output_size}px@{train_set.target_mpp}MPP"
    )
    return train_loader, val_loader, od_max


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, channels=3, base=32, residual_blocks=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, base, 7, bias=False),
            nn.InstanceNorm2d(base),
            nn.ReLU(inplace=True),
        ]
        in_channels = base
        for _ in range(2):
            out_channels = in_channels * 2
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
        layers += [ResidualBlock(in_channels) for _ in range(residual_blocks)]
        for _ in range(2):
            out_channels = in_channels // 2
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels, out_channels, 3, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(base, channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, channels=3, base=64):
        super().__init__()

        def block(in_channels, out_channels, stride=2, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(channels, base, normalize=False),
            *block(base, base * 2),
            *block(base * 2, base * 4),
            *block(base * 4, base * 8, stride=1),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)


def init_weights(module):
    classname = module.__class__.__name__
    if "Conv" in classname and hasattr(module, "weight"):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = int(pool_size)
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images.detach()
        returned = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                returned.append(image)
            elif random.random() > 0.5:
                index = random.randrange(self.pool_size)
                old = self.images[index].clone()
                self.images[index] = image.clone()
                returned.append(old)
            else:
                returned.append(image)
        return torch.cat(returned, dim=0)


def set_requires_grad(models, enabled):
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad = enabled


def denormalize(x):
    return ((x + 1) / 2).clamp(0, 1)


def simple_ssim(x, y, window_size=11):
    """Diagnostic raw paired SSIM only; never used as a training objective."""
    x, y = denormalize(x), denormalize(y)
    padding = window_size // 2
    mu_x = torch.nn.functional.avg_pool2d(x, window_size, 1, padding)
    mu_y = torch.nn.functional.avg_pool2d(y, window_size, 1, padding)
    sigma_x = torch.nn.functional.avg_pool2d(x * x, window_size, 1, padding) - mu_x.square()
    sigma_y = torch.nn.functional.avg_pool2d(y * y, window_size, 1, padding) - mu_y.square()
    sigma_xy = torch.nn.functional.avg_pool2d(x * y, window_size, 1, padding) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    )
    return score.mean()


def masked_l1(x, target, mask):
    mask = mask.to(dtype=x.dtype)
    channels = x.shape[1]
    return ((x - target).abs() * mask).sum() / (mask.sum() * channels + 1e-6)


def background_losses(real_a, real_b, fake_a, fake_b, params):
    """Unpaired, source-derived background constraints for both directions."""
    real_a_01 = denormalize(real_a)
    real_b_01 = denormalize(real_b)
    fake_a_01 = denormalize(fake_a)
    fake_b_01 = denormalize(fake_b)

    # Domain A contains the original OD direction: background is near 0 in [0, 1].
    a_od = real_a_01.mean(dim=1, keepdim=True)
    mask_a = (a_od < params["a_background_od_threshold"]).to(real_a.dtype)

    # Domain B background is bright and nearly achromatic RGB.
    b_brightness = real_b_01.mean(dim=1, keepdim=True)
    b_saturation = real_b_01.amax(dim=1, keepdim=True) - real_b_01.amin(dim=1, keepdim=True)
    mask_b = (
        (b_brightness > params["b_background_brightness_threshold"])
        & (b_saturation < params["b_background_saturation_threshold"])
    ).to(real_b.dtype)

    # Feather only the boundary; the masks are derived from real inputs and need no gradient.
    kernel = int(params.get("background_mask_blur_kernel", 5))
    if kernel > 1:
        padding = kernel // 2
        mask_a = F.avg_pool2d(mask_a, kernel, stride=1, padding=padding)
        mask_b = F.avg_pool2d(mask_b, kernel, stride=1, padding=padding)

    # A background (-1) must become white H&E (+1); B background (+1) must become OD zero (-1).
    loss_ab = masked_l1(fake_b_01, 1.0, mask_a)
    loss_ba = masked_l1(fake_a_01, 0.0, mask_b)
    return loss_ab, loss_ba, mask_a.mean(), mask_b.mean()


def weak_blurred_l1(fake, real, params):
    """Mildly registration-tolerant paired supervision in normalized space."""
    kernel = int(params["paired_blur_kernel"])
    if kernel % 2 == 0 or kernel < 1:
        raise ValueError("paired_blur_kernel must be a positive odd integer")
    sigma = float(params["paired_blur_sigma"])
    if kernel > 1 and sigma > 0:
        fake = TF.gaussian_blur(fake, [kernel, kernel], [sigma, sigma])
        real = TF.gaussian_blur(real, [kernel, kernel], [sigma, sigma])
    return F.l1_loss(fake, real)


class CycleGANTrainer:
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

        self.G_AB = ResNetGenerator(3, params["ngf"], params["residual_blocks"]).to(device)
        self.G_BA = ResNetGenerator(3, params["ngf"], params["residual_blocks"]).to(device)
        self.D_A = PatchDiscriminator(3, params["ndf"]).to(device)
        self.D_B = PatchDiscriminator(3, params["ndf"]).to(device)
        for model in (self.G_AB, self.G_BA, self.D_A, self.D_B):
            model.apply(init_weights)

        self.opt_g = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=params["lr"],
            betas=(params["beta1"], params["beta2"]),
        )
        self.opt_d_a = torch.optim.Adam(
            self.D_A.parameters(), lr=params["lr"], betas=(params["beta1"], params["beta2"])
        )
        self.opt_d_b = torch.optim.Adam(
            self.D_B.parameters(), lr=params["lr"], betas=(params["beta1"], params["beta2"])
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
        if params.get("resume_checkpoint"):
            self.load(params["resume_checkpoint"])

        print(
            f"G_AB={sum(p.numel() for p in self.G_AB.parameters()) / 1e6:.2f}M, "
            f"G_BA={sum(p.numel() for p in self.G_BA.parameters()) / 1e6:.2f}M"
        )

    def _generator_step(self, real_a, real_b):
        set_requires_grad([self.D_A, self.D_B], False)
        self.opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
            fake_b = self.G_AB(real_a)
            fake_a = self.G_BA(real_b)
            rec_a = self.G_BA(fake_b)
            rec_b = self.G_AB(fake_a)
            idt_a = self.G_BA(real_a)
            idt_b = self.G_AB(real_b)

            pred_fake_b = self.D_B(fake_b)
            pred_fake_a = self.D_A(fake_a)
            gan_ab = self.gan_loss(pred_fake_b, torch.ones_like(pred_fake_b))
            gan_ba = self.gan_loss(pred_fake_a, torch.ones_like(pred_fake_a))
            cycle_a = self.l1(rec_a, real_a)
            cycle_b = self.l1(rec_b, real_b)
            identity_a = self.l1(idt_a, real_a)
            identity_b = self.l1(idt_b, real_b)
            background_ab, background_ba, background_fraction_a, background_fraction_b = (
                background_losses(real_a, real_b, fake_a, fake_b, self.params)
            )
            if self.params.get("paired_training", False):
                paired_blur_ab = weak_blurred_l1(fake_b, real_b, self.params)
                paired_blur_ba = weak_blurred_l1(fake_a, real_a, self.params)
            else:
                paired_blur_ab = fake_b.new_zeros(())
                paired_blur_ba = fake_a.new_zeros(())
            loss_g = (
                gan_ab
                + gan_ba
                + self.params["lambda_cycle"] * (cycle_a + cycle_b)
                + self.params["lambda_identity"] * (identity_a + identity_b)
                + self.params["lambda_background"] * (background_ab + background_ba)
                + self.params.get("lambda_paired_blur", 0.0) * (paired_blur_ab + paired_blur_ba)
            )
        self.scaler_g.scale(loss_g).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()
        return {
            "G": loss_g.detach(),
            "gan_AB": gan_ab.detach(),
            "gan_BA": gan_ba.detach(),
            "cycle_A": cycle_a.detach(),
            "cycle_B": cycle_b.detach(),
            "identity_A": identity_a.detach(),
            "identity_B": identity_b.detach(),
            "background_AB": background_ab.detach(),
            "background_BA": background_ba.detach(),
            "background_fraction_A": background_fraction_a.detach(),
            "background_fraction_B": background_fraction_b.detach(),
            "paired_blur_AB": paired_blur_ab.detach(),
            "paired_blur_BA": paired_blur_ba.detach(),
        }, fake_a.detach(), fake_b.detach()

    def _discriminator_step(self, discriminator, optimizer, scaler, real, fake):
        set_requires_grad(discriminator, True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
            pred_real = discriminator(real)
            pred_fake = discriminator(fake)
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
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.params['num_epochs']}")
        for step, (real_a, real_b) in enumerate(pbar, start=1):
            real_a = real_a.to(self.device, non_blocking=True)
            real_b = real_b.to(self.device, non_blocking=True)
            g_metrics, fake_a, fake_b = self._generator_step(real_a, real_b)
            d_a = self._discriminator_step(
                self.D_A, self.opt_d_a, self.scaler_d_a, real_a, self.pool_a.query(fake_a)
            )
            d_b = self._discriminator_step(
                self.D_B, self.opt_d_b, self.scaler_d_b, real_b, self.pool_b.query(fake_b)
            )
            metrics = {**g_metrics, "D_A": d_a, "D_B": d_b}
            for key, value in metrics.items():
                totals[key] += float(value)
            pbar.set_postfix(
                G=f"{totals['G'] / step:.3f}",
                cycle=f"{(totals['cycle_A'] + totals['cycle_B']) / step:.3f}",
                bg=f"{(totals['background_AB'] + totals['background_BA']) / step:.3f}",
                pair=f"{(totals['paired_blur_AB'] + totals['paired_blur_BA']) / step:.3f}",
                D=f"{(totals['D_A'] + totals['D_B']) / (2 * step):.3f}",
            )
        return {key: value / len(self.train_loader) for key, value in totals.items()}

    @torch.no_grad()
    def validate(self):
        self.G_AB.eval()
        self.G_BA.eval()
        totals = defaultdict(float)
        count = 0
        preview = None
        for real_a, real_b in self.val_loader:
            real_a = real_a.to(self.device, non_blocking=True)
            real_b = real_b.to(self.device, non_blocking=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                fake_b = self.G_AB(real_a)
                fake_a = self.G_BA(real_b)
                rec_a = self.G_BA(fake_b)
                rec_b = self.G_AB(fake_a)
                cycle_a = self.l1(rec_a, real_a)
                cycle_b = self.l1(rec_b, real_b)
                raw_ssim = simple_ssim(fake_b.float(), real_b.float())
                background_ab, background_ba, background_fraction_a, background_fraction_b = (
                    background_losses(real_a, real_b, fake_a, fake_b, self.params)
                )
                if self.params.get("paired_training", False):
                    paired_blur_ab = weak_blurred_l1(fake_b, real_b, self.params)
                    paired_blur_ba = weak_blurred_l1(fake_a, real_a, self.params)
                else:
                    paired_blur_ab = fake_b.new_zeros(())
                    paired_blur_ba = fake_a.new_zeros(())
            batch = real_a.shape[0]
            totals["cycle_A"] += float(cycle_a) * batch
            totals["cycle_B"] += float(cycle_b) * batch
            totals["raw_paired_ssim"] += float(raw_ssim) * batch
            totals["background_AB"] += float(background_ab) * batch
            totals["background_BA"] += float(background_ba) * batch
            totals["background_fraction_A"] += float(background_fraction_a) * batch
            totals["background_fraction_B"] += float(background_fraction_b) * batch
            totals["paired_blur_AB"] += float(paired_blur_ab) * batch
            totals["paired_blur_BA"] += float(paired_blur_ba) * batch
            count += batch
            if preview is None:
                preview = tuple(x.detach().float().cpu() for x in (real_a, fake_b, real_b, rec_a, real_b, fake_a, real_a, rec_b))
        metrics = {key: value / count for key, value in totals.items()}
        metrics["cycle_total"] = metrics["cycle_A"] + metrics["cycle_B"]
        metrics["paired_blur_total"] = metrics["paired_blur_AB"] + metrics["paired_blur_BA"]
        metrics["selection_score"] = (
            self.params["lambda_cycle"] * metrics["cycle_total"]
            + self.params.get("lambda_paired_blur", 0.0) * metrics["paired_blur_total"]
            + self.params["lambda_background"]
            * (metrics["background_AB"] + metrics["background_BA"])
        )
        return metrics, preview

    def save_preview(self, epoch, preview):
        labels = [
            "Unstain OD (A)", "Fake H&E (A→B)", "Paired Real H&E", "Recovered A",
            "Real H&E (B)", "Fake Unstain (B→A)", "Paired Real A", "Recovered B",
        ]
        rows = min(self.params["preview_count"], preview[0].shape[0])
        fig, axes = plt.subplots(rows, 8, figsize=(24, 3 * rows), squeeze=False)
        for row in range(rows):
            for col, batch in enumerate(preview):
                image = denormalize(batch[row]).permute(1, 2, 0).numpy()
                axes[row, col].imshow(image.clip(0, 1), cmap="gray" if col in (0, 3, 5, 6) else None)
                axes[row, col].set_title(labels[col])
                axes[row, col].axis("off")
        fig.suptitle(f"Epoch {epoch + 1} — paired CycleGAN at {self.params['target_mpp']} MPP", y=1.01)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"epoch_{epoch + 1:04d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def checkpoint(self, epoch, is_best=False):
        state = {
            "epoch": epoch,
            "G_AB": self.G_AB.state_dict(),
            "G_BA": self.G_BA.state_dict(),
            "D_A": self.D_A.state_dict(),
            "D_B": self.D_B.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d_a": self.opt_d_a.state_dict(),
            "opt_d_b": self.opt_d_b.state_dict(),
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
            "scaler_g": self.scaler_g.state_dict(),
            "scaler_d_a": self.scaler_d_a.state_dict(),
            "scaler_d_b": self.scaler_d_b.state_dict(),
            "od_max": self.od_max,
            "params": self.params,
            "history": self.history,
            "best_score": self.best_score,
        }
        torch.save(state, self.checkpoint_dir / "latest.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")
        if (epoch + 1) % self.params["save_every"] == 0:
            torch.save(state, self.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt")

    def load(self, checkpoint_path):
        # This is a trusted, locally generated full training checkpoint containing
        # optimizer state and Path-valued parameters, not a weights-only artifact.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        for name in ("G_AB", "G_BA", "D_A", "D_B"):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.opt_g.load_state_dict(checkpoint["opt_g"])
        self.opt_d_a.load_state_dict(checkpoint["opt_d_a"])
        self.opt_d_b.load_state_dict(checkpoint["opt_d_b"])
        for scheduler, state in zip(self.schedulers, checkpoint["schedulers"]):
            scheduler.load_state_dict(state)
        for scaler_name in ("scaler_g", "scaler_d_a", "scaler_d_b"):
            if scaler_name in checkpoint:
                getattr(self, scaler_name).load_state_dict(checkpoint[scaler_name])
        self.start_epoch = checkpoint["epoch"] + 1
        self.history = checkpoint.get("history", [])
        self.best_score = checkpoint.get("best_score", checkpoint.get("best_cycle", math.inf))
        print(f"Resumed from epoch {self.start_epoch}")

    def fit(self):
        for epoch in range(self.start_epoch, self.params["num_epochs"]):
            train_metrics = self.train_epoch(epoch)
            val_metrics, preview = self.validate()
            for scheduler in self.schedulers:
                scheduler.step()
            row = {
                "epoch": epoch + 1,
                "lr": self.opt_g.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
            self.history.append(row)
            is_best = val_metrics["selection_score"] < self.best_score
            if is_best:
                self.best_score = val_metrics["selection_score"]
            self.checkpoint(epoch, is_best)
            self.save_preview(epoch, preview)
            print("train:", {k: round(v, 4) for k, v in train_metrics.items()})
            print("val (diagnostic):", {k: round(v, 4) for k, v in val_metrics.items()})
            if is_best:
                print(f"new best paired-cycle checkpoint: epoch {epoch + 1}")
            with (self.output_dir / "history.json").open("w") as file:
                json.dump(self.history, file, indent=2)
