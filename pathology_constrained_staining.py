from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from cyclegan_core import (
    ImagePool,
    PatchDiscriminator,
    ResNetGenerator,
    denormalize,
    init_weights,
    masked_l1,
    set_requires_grad,
    simple_ssim,
)
from two_stage_virtual_staining import (
    ConvBlock,
    DownBlock,
    UpBlock,
    rgb_gradient_loss,
    rgb_laplacian_loss,
)


# Columns are the conventional optical-density directions for hematoxylin and eosin.
DEFAULT_HE_BASIS = torch.tensor(
    [[0.650, 0.072], [0.704, 0.990], [0.286, 0.105]], dtype=torch.float32
)
DEFAULT_HE_BASIS = F.normalize(DEFAULT_HE_BASIS, dim=0)
DEFAULT_HE_INVERSE = torch.linalg.pinv(DEFAULT_HE_BASIS)


class AxialAttentionBlock(nn.Module):
    """Global row/column context with O(HW(H+W)) rather than full O((HW)^2)."""

    def __init__(self, channels, heads=8, dropout=0.1, mlp_ratio=2):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.row_norm = nn.LayerNorm(channels)
        self.col_norm = nn.LayerNorm(channels)
        self.row_attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        self.col_attention = nn.MultiheadAttention(
            channels, heads, dropout=dropout, batch_first=True
        )
        hidden = channels * mlp_ratio
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, channels), nn.Dropout(dropout),
        )

    def forward(self, feature):
        batch, channels, height, width = feature.shape
        x = feature.permute(0, 2, 3, 1)

        rows = x.reshape(batch * height, width, channels)
        normalized = self.row_norm(rows)
        rows = rows + self.row_attention(normalized, normalized, normalized, need_weights=False)[0]
        x = rows.reshape(batch, height, width, channels)

        columns = x.permute(0, 2, 1, 3).reshape(batch * width, height, channels)
        normalized = self.col_norm(columns)
        columns = columns + self.col_attention(
            normalized, normalized, normalized, need_weights=False
        )[0]
        x = columns.reshape(batch, width, height, channels).permute(0, 2, 1, 3)
        x = x + self.ffn(self.ffn_norm(x))
        return x.permute(0, 3, 1, 2).contiguous()


class PathologyStainDecoder(nn.Module):
    """Predicted H&E OD -> non-negative H/E concentrations -> physical RGB."""

    def __init__(
        self,
        base=32,
        attention_blocks=2,
        attention_heads=8,
        attention_dropout=0.1,
        max_concentration=4.0,
        stain_basis_delta=0.10,
    ):
        super().__init__()
        self.input_channels = 1
        self.max_concentration = float(max_concentration)
        self.stain_basis_delta = float(stain_basis_delta)

        self.e1 = ConvBlock(1, base)
        self.e2 = DownBlock(base, base * 2)
        self.e3 = DownBlock(base * 2, base * 4)
        self.e4 = DownBlock(base * 4, base * 8)
        self.bottleneck = nn.Sequential(
            DownBlock(base * 8, base * 8),
            ConvBlock(base * 8, base * 8),
        )
        self.context = nn.Sequential(
            *[
                AxialAttentionBlock(
                    base * 8, attention_heads, attention_dropout, mlp_ratio=2
                )
                for _ in range(attention_blocks)
            ]
        )
        self.u4 = UpBlock(base * 8, base * 8, base * 8, refine=True)
        self.u3 = UpBlock(base * 8, base * 4, base * 4, refine=True)
        self.u2 = UpBlock(base * 4, base * 2, base * 2, refine=True)
        self.u1 = UpBlock(base * 2, base, base, refine=True)
        self.output_refine = ConvBlock(base, base)
        self.output_head = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(base, 3, 3)
        )

        self.register_buffer("reference_stain_basis", DEFAULT_HE_BASIS.clone())
        self.stain_basis_offset = nn.Parameter(torch.zeros_like(DEFAULT_HE_BASIS))

    def initialize_output(self):
        final = self.output_head[-1]
        nn.init.normal_(final.weight, 0.0, 0.02)
        with torch.no_grad():
            final.bias[:2].fill_(-2.0)
            final.bias[2].fill_(-1.5)

    def stain_basis(self):
        offset = self.stain_basis_delta * torch.tanh(self.stain_basis_offset)
        return F.normalize((self.reference_stain_basis + offset).clamp_min(1e-4), dim=0)

    def concentrations_to_rgb(self, concentrations):
        optical_density = torch.einsum(
            "ck,bkhw->bchw", self.stain_basis(), concentrations
        )
        rgb_01 = torch.exp(-optical_density).clamp(0.0, 1.0)
        return rgb_01 * 2.0 - 1.0

    def forward(self, predicted_hne_od):
        e1 = self.e1(predicted_hne_od)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        bottleneck = self.context(self.bottleneck(e4))
        decoded = self.u1(
            self.u2(self.u3(self.u4(bottleneck, e4), e3), e2), e1
        )
        raw = self.output_head(self.output_refine(decoded))
        concentrations = self.max_concentration * torch.sigmoid(raw[:, :2])
        log_scale = 3.0 * torch.tanh(raw[:, 2:3])
        rgb = self.concentrations_to_rgb(concentrations)
        return {
            "rgb": rgb,
            "concentrations": concentrations,
            "log_scale": log_scale,
            "uncertainty": torch.exp(log_scale),
        }


class CellAwareStructureGenerator(nn.Module):
    """Unstain OD -> H&E OD structure and calibrated per-pixel uncertainty."""

    def __init__(self, base=32, attention_blocks=2, attention_heads=8, dropout=0.1):
        super().__init__()
        self.e1 = ConvBlock(1, base)
        self.e2 = DownBlock(base, base * 2)
        self.e3 = DownBlock(base * 2, base * 4)
        self.e4 = DownBlock(base * 4, base * 8)
        self.bottleneck = nn.Sequential(
            DownBlock(base * 8, base * 8), ConvBlock(base * 8, base * 8)
        )
        self.context = nn.Sequential(
            *[
                AxialAttentionBlock(base * 8, attention_heads, dropout, mlp_ratio=2)
                for _ in range(attention_blocks)
            ]
        )
        self.u4 = UpBlock(base * 8, base * 8, base * 8, refine=True)
        self.u3 = UpBlock(base * 8, base * 4, base * 4, refine=True)
        self.u2 = UpBlock(base * 4, base * 2, base * 2, refine=True)
        self.u1 = UpBlock(base * 2, base, base, refine=True)
        self.head = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(base, 2, 3))

    def initialize_output(self):
        final = self.head[-1]
        nn.init.normal_(final.weight, 0.0, 0.02)
        with torch.no_grad():
            final.bias[0].zero_()
            final.bias[1].fill_(-1.5)

    def forward(self, unstain_od):
        e1 = self.e1(unstain_od)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        x = self.context(self.bottleneck(e4))
        x = self.u1(self.u2(self.u3(self.u4(x, e4), e3), e2), e1)
        raw = self.head(x)
        log_scale = 3.0 * torch.tanh(raw[:, 1:2])
        return {
            "od": torch.tanh(raw[:, :1]),
            "log_scale": log_scale,
            "uncertainty": torch.exp(log_scale),
        }


def structure_od(model_output):
    return model_output["od"] if isinstance(model_output, dict) else model_output


def soft_nuclei_map(hne_od, threshold=0.50, temperature=0.08):
    od_01 = denormalize(hne_od)
    return torch.sigmoid((od_01 - float(threshold)) / float(temperature))


def soft_dice_loss(predicted, target, epsilon=1e-6):
    numerator = 2.0 * (predicted * target).sum(dim=(1, 2, 3)) + epsilon
    denominator = (
        predicted.square().sum(dim=(1, 2, 3))
        + target.square().sum(dim=(1, 2, 3))
        + epsilon
    )
    return (1.0 - numerator / denominator).mean()


def cell_density_loss(predicted_map, target_map, grid_size=16):
    predicted_density = F.adaptive_avg_pool2d(predicted_map, (grid_size, grid_size))
    target_density = F.adaptive_avg_pool2d(target_map, (grid_size, grid_size))
    return F.l1_loss(predicted_density, target_density)


def _overlap_for_shift(predicted, target, dy, dx):
    height, width = predicted.shape[-2:]
    pred_y = slice(max(0, dy), min(height, height + dy))
    pred_x = slice(max(0, dx), min(width, width + dx))
    target_y = slice(max(0, -dy), min(height, height - dy))
    target_x = slice(max(0, -dx), min(width, width - dx))
    return predicted[..., pred_y, pred_x], target[..., target_y, target_x]


def shift_tolerant_pair_losses(predicted, target, radius=4, step=2):
    """Choose one small whole-patch shift; avoids averaging over registration error."""
    offsets = range(-int(radius), int(radius) + 1, max(1, int(step)))
    with torch.no_grad():
        candidates = []
        for dy in offsets:
            for dx in offsets:
                predicted_crop, target_crop = _overlap_for_shift(
                    predicted, target, dy, dx
                )
                candidates.append(
                    (float(F.l1_loss(predicted_crop, target_crop)), dy, dx)
                )
        _, best_dy, best_dx = min(candidates)
    predicted_crop, target_crop = _overlap_for_shift(
        predicted, target, best_dy, best_dx
    )
    l1 = F.l1_loss(predicted_crop, target_crop)
    ssim = simple_ssim(predicted_crop, target_crop)
    return {
        "objective": l1 + 0.25 * (1.0 - ssim), "l1": l1, "ssim": ssim,
        "dy": best_dy, "dx": best_dx,
    }


def rgb_to_he_concentrations(rgb_normalized, maximum=4.0):
    """Fixed-reference H&E color deconvolution used only to construct loss targets."""
    output_dtype = rgb_normalized.dtype
    with torch.amp.autocast(rgb_normalized.device.type, enabled=False):
        rgb_01 = denormalize(rgb_normalized.float()).clamp(1 / 255, 1.0)
        optical_density = -torch.log(rgb_01)
        inverse = DEFAULT_HE_INVERSE.to(device=rgb_normalized.device)
        concentrations = torch.einsum("kc,bchw->bkhw", inverse, optical_density)
    return concentrations.clamp(0.0, float(maximum)).to(output_dtype)


def stain_moment_loss(predicted, target, tissue_threshold=0.05):
    mask = (target.sum(dim=1, keepdim=True) > tissue_threshold).to(target.dtype)
    denominator = mask.sum(dim=(0, 2, 3)).clamp_min(1.0)

    def moments(value):
        mean = (value * mask).sum(dim=(0, 2, 3)) / denominator
        variance = ((value - mean[None, :, None, None]) ** 2 * mask).sum(
            dim=(0, 2, 3)
        ) / denominator
        return mean, torch.sqrt(variance + 1e-6)

    predicted_mean, predicted_std = moments(predicted)
    target_mean, target_std = moments(target)
    return F.l1_loss(predicted_mean, target_mean) + F.l1_loss(
        predicted_std, target_std
    )


def laplace_uncertainty_loss(fake_rgb, real_rgb, log_scale):
    error = (fake_rgb - real_rgb).abs().mean(dim=1, keepdim=True)
    return (error * torch.exp(-log_scale) + log_scale).mean(), error


def uncertainty_calibration_loss(uncertainty, detached_error):
    return F.smooth_l1_loss(uncertainty, detached_error.detach().clamp_min(1e-3))


def uncertainty_error_correlation(uncertainty, error):
    x = uncertainty.flatten(1)
    y = error.flatten(1)
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    numerator = (x * y).mean(dim=1)
    denominator = x.square().mean(dim=1).sqrt() * y.square().mean(dim=1).sqrt()
    return (numerator / denominator.clamp_min(1e-8)).mean()


class CellAwareStructureTrainer:
    """Bidirectional CycleGAN whose forward model explicitly learns nuclear structure."""

    def __init__(self, params, train_loader, val_loader, unstain_od_max, device):
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.unstain_od_max = float(unstain_od_max)
        self.device = device
        self.output_dir = Path(params["output_dir"])
        self.checkpoint_dir = Path(params["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.G_AB = CellAwareStructureGenerator(
            base=params["base_channels"],
            attention_blocks=params["attention_blocks"],
            attention_heads=params["attention_heads"],
            dropout=params["attention_dropout"],
        ).to(device)
        self.G_BA = ResNetGenerator(
            1, params["reverse_ngf"], params["reverse_residual_blocks"]
        ).to(device)
        self.D_A = PatchDiscriminator(1, params["ndf"]).to(device)
        self.D_B = PatchDiscriminator(1, params["ndf"]).to(device)
        for model in (self.G_AB, self.G_BA, self.D_A, self.D_B):
            model.apply(init_weights)
        self.G_AB.initialize_output()

        self.opt_g = torch.optim.AdamW(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=params["lr_g"], betas=(params["beta1"], params["beta2"]),
            weight_decay=params["weight_decay"],
        )
        self.opt_d_a = torch.optim.Adam(
            self.D_A.parameters(), lr=params["lr_d"],
            betas=(params["beta1"], params["beta2"]),
        )
        self.opt_d_b = torch.optim.Adam(
            self.D_B.parameters(), lr=params["lr_d"],
            betas=(params["beta1"], params["beta2"]),
        )
        self.schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, params["num_epochs"])
            for optimizer in (self.opt_g, self.opt_d_a, self.opt_d_b)
        ]
        self.gan_loss = nn.MSELoss()
        self.pool_a = ImagePool(params["pool_size"])
        self.pool_b = ImagePool(params["pool_size"])
        self.amp_enabled = device.type == "cuda"
        self.scaler_g = torch.amp.GradScaler(device=device.type, enabled=self.amp_enabled)
        self.scaler_d_a = torch.amp.GradScaler(device=device.type, enabled=self.amp_enabled)
        self.scaler_d_b = torch.amp.GradScaler(device=device.type, enabled=self.amp_enabled)
        self.history = []
        self.best_score = math.inf

    def background_losses(self, real_a, real_b, fake_a, fake_b):
        mask_a = (denormalize(real_a) < self.params["a_background_od_threshold"]).to(real_a.dtype)
        mask_b = (denormalize(real_b) < self.params["b_background_od_threshold"]).to(real_b.dtype)
        loss_ab = masked_l1(denormalize(fake_b), 0.0, mask_a)
        loss_ba = masked_l1(denormalize(fake_a), 0.0, mask_b)
        return loss_ab, loss_ba

    def generator_step(self, real_a, real_b):
        set_requires_grad([self.D_A, self.D_B], False)
        self.opt_g.zero_grad(set_to_none=True)
        with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
            forward_output = self.G_AB(real_a)
            fake_b = forward_output["od"]
            fake_a = self.G_BA(real_b)
            rec_a = self.G_BA(fake_b)
            rec_b = self.G_AB(fake_a)["od"]
            identity_a = self.G_BA(real_a)
            identity_b = self.G_AB(real_b)["od"]

            pred_fake_b = self.D_B(fake_b)
            pred_fake_a = self.D_A(fake_a)
            gan_ab = self.gan_loss(pred_fake_b, torch.ones_like(pred_fake_b))
            gan_ba = self.gan_loss(pred_fake_a, torch.ones_like(pred_fake_a))
            cycle_a = F.l1_loss(rec_a, real_a)
            cycle_b = F.l1_loss(rec_b, real_b)
            identity_a_loss = F.l1_loss(identity_a, real_a)
            identity_b_loss = F.l1_loss(identity_b, real_b)

            paired = shift_tolerant_pair_losses(
                fake_b, real_b,
                radius=self.params["registration_shift_radius"],
                step=self.params["registration_shift_step"],
            )
            aligned_fake, aligned_real = _overlap_for_shift(
                fake_b, real_b, paired["dy"], paired["dx"]
            )
            aligned_log_scale, _ = _overlap_for_shift(
                forward_output["log_scale"], real_b, paired["dy"], paired["dx"]
            )
            predicted_nuclei = soft_nuclei_map(
                aligned_fake, self.params["nucleus_od_threshold"],
                self.params["nucleus_temperature"],
            )
            target_nuclei = soft_nuclei_map(
                aligned_real, self.params["nucleus_od_threshold"],
                self.params["nucleus_temperature"],
            )
            nuclei = soft_dice_loss(predicted_nuclei, target_nuclei) + F.l1_loss(
                predicted_nuclei, target_nuclei
            )
            density = cell_density_loss(
                predicted_nuclei, target_nuclei, self.params["cell_density_grid"]
            )
            gradient = rgb_gradient_loss(aligned_fake, aligned_real)
            laplacian = rgb_laplacian_loss(aligned_fake, aligned_real)
            uncertainty_nll, error_map = laplace_uncertainty_loss(
                aligned_fake, aligned_real, aligned_log_scale
            )
            uncertainty = torch.exp(aligned_log_scale)
            uncertainty_calibration = uncertainty_calibration_loss(uncertainty, error_map)
            background_ab, background_ba = self.background_losses(
                real_a, real_b, fake_a, fake_b
            )

            total = (
                self.params["lambda_gan"] * (gan_ab + gan_ba)
                + self.params["lambda_cycle"] * (cycle_a + cycle_b)
                + self.params["lambda_identity"] * (identity_a_loss + identity_b_loss)
                + self.params["lambda_paired"] * paired["l1"]
                + self.params["lambda_ssim"] * (1.0 - paired["ssim"])
                + self.params["lambda_gradient"] * gradient
                + self.params["lambda_laplacian"] * laplacian
                + self.params["lambda_nuclei"] * nuclei
                + self.params["lambda_cell_density"] * density
                + self.params["lambda_uncertainty"] * uncertainty_nll
                + self.params["lambda_uncertainty_calibration"] * uncertainty_calibration
                + self.params["lambda_background"] * (background_ab + background_ba)
            )
        self.scaler_g.scale(total).backward()
        self.scaler_g.unscale_(self.opt_g)
        torch.nn.utils.clip_grad_norm_(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            self.params["grad_clip"],
        )
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()
        metrics = {
            "G": total, "gan_AB": gan_ab, "gan_BA": gan_ba,
            "cycle_A": cycle_a, "cycle_B": cycle_b,
            "identity_A": identity_a_loss, "identity_B": identity_b_loss,
            "paired_l1": paired["l1"], "paired_ssim": paired["ssim"],
            "gradient": gradient, "laplacian": laplacian,
            "nuclei": nuclei, "cell_density": density,
            "uncertainty_nll": uncertainty_nll,
            "uncertainty_calibration": uncertainty_calibration,
            "background_AB": background_ab, "background_BA": background_ba,
            "registration_shift": fake_b.new_tensor(
                math.hypot(paired["dy"], paired["dx"])
            ),
        }
        return {key: value.detach() for key, value in metrics.items()}, fake_a.detach(), fake_b.detach()

    def discriminator_step(self, discriminator, optimizer, scaler, real, fake):
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
        pbar = tqdm(
            self.train_loader,
            desc=f"Cell structure epoch {epoch + 1}/{self.params['num_epochs']}",
        )
        for step, (real_a, real_b) in enumerate(pbar, start=1):
            real_a = real_a.to(self.device, non_blocking=True)
            real_b = real_b.to(self.device, non_blocking=True)
            metrics, fake_a, fake_b = self.generator_step(real_a, real_b)
            metrics["D_A"] = self.discriminator_step(
                self.D_A, self.opt_d_a, self.scaler_d_a,
                real_a, self.pool_a.query(fake_a),
            )
            metrics["D_B"] = self.discriminator_step(
                self.D_B, self.opt_d_b, self.scaler_d_b,
                real_b, self.pool_b.query(fake_b),
            )
            for key, value in metrics.items():
                totals[key] += float(value)
            pbar.set_postfix(
                G=f"{totals['G'] / step:.3f}",
                D=f"{(totals['D_A'] + totals['D_B']) / (2 * step):.3f}",
                SSIM=f"{totals['paired_ssim'] / step:.3f}",
                nuclei=f"{totals['nuclei'] / step:.3f}",
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
            output = self.G_AB(real_a)
            fake_b = output["od"]
            rec_a = self.G_BA(fake_b)
            paired = shift_tolerant_pair_losses(
                fake_b, real_b,
                self.params["registration_shift_radius"],
                self.params["registration_shift_step"],
            )
            aligned_fake, aligned_real = _overlap_for_shift(
                fake_b, real_b, paired["dy"], paired["dx"]
            )
            aligned_uncertainty, _ = _overlap_for_shift(
                output["uncertainty"], real_b, paired["dy"], paired["dx"]
            )
            predicted_nuclei = soft_nuclei_map(
                aligned_fake, self.params["nucleus_od_threshold"],
                self.params["nucleus_temperature"],
            )
            target_nuclei = soft_nuclei_map(
                aligned_real, self.params["nucleus_od_threshold"],
                self.params["nucleus_temperature"],
            )
            error_map = (aligned_fake - aligned_real).abs()
            metrics = {
                "paired_l1": paired["l1"], "paired_ssim": paired["ssim"],
                "raw_ssim": simple_ssim(fake_b, real_b),
                "gradient": rgb_gradient_loss(aligned_fake, aligned_real),
                "laplacian": rgb_laplacian_loss(aligned_fake, aligned_real),
                "nuclei": soft_dice_loss(predicted_nuclei, target_nuclei),
                "cell_density": cell_density_loss(
                    predicted_nuclei, target_nuclei, self.params["cell_density_grid"]
                ),
                "cycle_A": F.l1_loss(rec_a, real_a),
                "uncertainty_error_corr": uncertainty_error_correlation(
                    aligned_uncertainty, error_map
                ),
            }
            batch = real_a.shape[0]
            for key, value in metrics.items():
                totals[key] += float(value) * batch
            count += batch
            if preview is None:
                preview = tuple(
                    value.detach().float().cpu()
                    for value in (
                        real_a, fake_b, real_b, predicted_nuclei, target_nuclei,
                        output["uncertainty"], (fake_b - real_b).abs(), rec_a,
                    )
                )
        result = {key: value / count for key, value in totals.items()}
        result["selection_score"] = (
            result["paired_l1"] + (1.0 - result["paired_ssim"])
            + 0.5 * result["nuclei"] + 0.25 * result["cell_density"]
            + 0.25 * result["gradient"] + 0.10 * result["laplacian"]
        )
        return result, preview

    def save_preview(self, epoch, preview):
        labels = [
            "Unstain OD", "Predicted H&E OD", "Real H&E OD",
            "Predicted nuclei response", "Target nuclei response",
            "Uncertainty", "Absolute OD error", "Recovered Unstain OD",
        ]
        rows = min(self.params["preview_count"], preview[0].shape[0])
        fig, axes = plt.subplots(rows, len(labels), figsize=(24, 3 * rows), squeeze=False)
        for row in range(rows):
            for column, batch in enumerate(preview):
                panel = batch[row, 0]
                if column in {0, 1, 2, 7}:
                    panel = denormalize(panel)
                    cmap = "gray"
                else:
                    cmap = "magma"
                axes[row, column].imshow(panel.numpy(), cmap=cmap)
                axes[row, column].set_title(labels[column])
                axes[row, column].axis("off")
        fig.suptitle(f"Cell-aware structure completion — epoch {epoch + 1}", y=1.01)
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
            "scaler_g": self.scaler_g.state_dict(),
            "scaler_d_a": self.scaler_d_a.state_dict(),
            "scaler_d_b": self.scaler_d_b.state_dict(),
            "history": self.history, "best_score": self.best_score,
            "od_max": self.unstain_od_max, "params": self.params,
            "architecture": "CellAwareStructureGenerator",
        }
        torch.save(state, self.checkpoint_dir / "latest.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")
        if (epoch + 1) % self.params["save_every"] == 0:
            torch.save(state, self.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt")

    def fit(self):
        for epoch in range(self.params["num_epochs"]):
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
            print("train:", {key: round(value, 4) for key, value in train_metrics.items()})
            print("val:", {key: round(value, 4) for key, value in val_metrics.items()})
            if is_best:
                print(f"new best cell-aware Stage 1: epoch {epoch + 1}")
            with (self.output_dir / "history.json").open("w") as file:
                json.dump(self.history, file, indent=2)


def load_cell_aware_structure_generator(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = state["params"]
    model = CellAwareStructureGenerator(
        base=params["base_channels"],
        attention_blocks=params["attention_blocks"],
        attention_heads=params["attention_heads"],
        dropout=params["attention_dropout"],
    )
    model.load_state_dict(state["G_AB"])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model, state


def load_frozen_structure_generator(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = state["params"]
    model = ResNetGenerator(1, params["ngf"], params["residual_blocks"])
    model.load_state_dict(state["G_AB"])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    metadata = {
        "epoch": int(state["epoch"]) + 1,
        "od_max": float(state["od_max"]),
        "input_size": int(params["input_size"]),
        "source_mpp": float(params["source_mpp"]),
        "target_mpp": float(params["target_mpp"]),
    }
    return model, metadata


class PathologyColorizerTrainer:
    def __init__(self, params, train_loader, val_loader, structure_generator, device):
        self.params = params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.structure_generator = structure_generator.eval()
        self.device = device
        self.output_dir = Path(params["output_dir"])
        self.checkpoint_dir = Path(params["checkpoint_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.G = PathologyStainDecoder(
            base=params["base_channels"],
            attention_blocks=params["attention_blocks"],
            attention_heads=params["attention_heads"],
            attention_dropout=params["attention_dropout"],
            max_concentration=params["max_concentration"],
            stain_basis_delta=params["stain_basis_delta"],
        ).to(device)
        self.D = PatchDiscriminator(4, params["ndf"]).to(device)
        self.G.apply(init_weights)
        self.G.initialize_output()
        self.D.apply(init_weights)

        self.opt_g = torch.optim.AdamW(
            self.G.parameters(), lr=params["lr_g"],
            betas=(params["beta1"], params["beta2"]),
            weight_decay=params["weight_decay"],
        )
        self.opt_d = torch.optim.Adam(
            self.D.parameters(), lr=params["lr_d"],
            betas=(params["beta1"], params["beta2"]),
        )
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_g, params["num_epochs"]
        )
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_d, params["num_epochs"]
        )
        self.gan_loss = nn.MSELoss()
        self.amp_enabled = device.type == "cuda"
        self.scaler_g = torch.amp.GradScaler(device=device.type, enabled=self.amp_enabled)
        self.scaler_d = torch.amp.GradScaler(device=device.type, enabled=self.amp_enabled)
        self.history = []
        self.best_score = math.inf

    def background_loss(self, predicted_od, fake_rgb):
        predicted_od_01 = denormalize(predicted_od)
        mask = (predicted_od_01 < self.params["background_od_threshold"]).to(fake_rgb.dtype)
        return masked_l1(denormalize(fake_rgb), 1.0, mask)

    def generator_losses(self, predicted_od, real_rgb):
        output = self.G(predicted_od)
        fake_rgb = output["rgb"]
        predicted_concentrations = output["concentrations"]
        target_concentrations = rgb_to_he_concentrations(
            real_rgb, maximum=self.params["max_concentration"]
        )
        discriminator_prediction = self.D(torch.cat([predicted_od, fake_rgb], dim=1))

        gan = self.gan_loss(
            discriminator_prediction, torch.ones_like(discriminator_prediction)
        )
        rgb_l1 = F.l1_loss(fake_rgb, real_rgb)
        ssim_loss = 1.0 - simple_ssim(fake_rgb, real_rgb)
        gradient = rgb_gradient_loss(fake_rgb, real_rgb)
        laplacian = rgb_laplacian_loss(fake_rgb, real_rgb)
        h_l1 = F.l1_loss(predicted_concentrations[:, :1], target_concentrations[:, :1])
        e_l1 = F.l1_loss(predicted_concentrations[:, 1:], target_concentrations[:, 1:])
        stain_moments = stain_moment_loss(predicted_concentrations, target_concentrations)
        h_morphology = (
            rgb_gradient_loss(
                predicted_concentrations[:, :1], target_concentrations[:, :1]
            )
            + rgb_laplacian_loss(
                predicted_concentrations[:, :1], target_concentrations[:, :1]
            )
        )
        uncertainty_nll, error_map = laplace_uncertainty_loss(
            fake_rgb, real_rgb, output["log_scale"]
        )
        uncertainty_calibration = uncertainty_calibration_loss(
            output["uncertainty"], error_map
        )
        background = self.background_loss(predicted_od, fake_rgb)
        basis_regularization = self.G.stain_basis_offset.square().mean()

        total = (
            self.params["lambda_gan"] * gan
            + self.params["lambda_rgb"] * rgb_l1
            + self.params["lambda_ssim"] * ssim_loss
            + self.params["lambda_gradient"] * gradient
            + self.params["lambda_laplacian"] * laplacian
            + self.params["lambda_h"] * h_l1
            + self.params["lambda_e"] * e_l1
            + self.params["lambda_stain_moments"] * stain_moments
            + self.params["lambda_h_morphology"] * h_morphology
            + self.params["lambda_uncertainty"] * uncertainty_nll
            + self.params["lambda_uncertainty_calibration"] * uncertainty_calibration
            + self.params["lambda_background"] * background
            + self.params["lambda_stain_basis"] * basis_regularization
        )
        losses = {
            "G": total, "gan": gan, "rgb_l1": rgb_l1,
            "ssim_loss": ssim_loss, "ssim_score": 1.0 - ssim_loss,
            "gradient": gradient, "laplacian": laplacian,
            "h_l1": h_l1, "e_l1": e_l1, "stain_moments": stain_moments,
            "h_morphology": h_morphology, "uncertainty_nll": uncertainty_nll,
            "uncertainty_calibration": uncertainty_calibration,
            "background": background, "basis_regularization": basis_regularization,
        }
        return output, target_concentrations, error_map, losses

    def train_epoch(self, epoch):
        self.G.train(); self.D.train(); self.structure_generator.eval()
        totals = defaultdict(float)
        pbar = tqdm(
            self.train_loader,
            desc=f"Pathology color epoch {epoch + 1}/{self.params['num_epochs']}",
        )
        for step, (unstain_od, real_rgb) in enumerate(pbar, start=1):
            unstain_od = unstain_od.to(self.device, non_blocking=True)
            real_rgb = real_rgb.to(self.device, non_blocking=True)
            with torch.no_grad():
                predicted_od = structure_od(self.structure_generator(unstain_od))

            for parameter in self.D.parameters():
                parameter.requires_grad = False
            self.opt_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                output, _, _, losses = self.generator_losses(predicted_od, real_rgb)
            self.scaler_g.scale(losses["G"]).backward()
            self.scaler_g.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.params["grad_clip"])
            self.scaler_g.step(self.opt_g)
            self.scaler_g.update()

            for parameter in self.D.parameters():
                parameter.requires_grad = True
            self.opt_d.zero_grad(set_to_none=True)
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                pred_real = self.D(torch.cat([predicted_od, real_rgb], dim=1))
                pred_fake = self.D(torch.cat([predicted_od, output["rgb"].detach()], dim=1))
                loss_d = 0.5 * (
                    self.gan_loss(pred_real, torch.ones_like(pred_real))
                    + self.gan_loss(pred_fake, torch.zeros_like(pred_fake))
                )
            self.scaler_d.scale(loss_d).backward()
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()

            losses["D"] = loss_d
            for key, value in losses.items():
                totals[key] += float(value.detach())
            pbar.set_postfix(
                G=f"{totals['G'] / step:.3f}",
                D=f"{totals['D'] / step:.3f}",
                SSIM=f"{totals['ssim_score'] / step:.3f}",
                H=f"{totals['h_l1'] / step:.3f}",
            )
        return {key: value / len(self.train_loader) for key, value in totals.items()}

    @torch.no_grad()
    def validate(self):
        self.G.eval(); self.D.eval(); self.structure_generator.eval()
        totals = defaultdict(float)
        count = 0
        preview = None
        for unstain_od, real_rgb in self.val_loader:
            unstain_od = unstain_od.to(self.device, non_blocking=True)
            real_rgb = real_rgb.to(self.device, non_blocking=True)
            predicted_od = structure_od(self.structure_generator(unstain_od))
            with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                output, target_concentrations, error_map, losses = self.generator_losses(
                    predicted_od, real_rgb
                )
            mse = F.mse_loss(denormalize(output["rgb"]), denormalize(real_rgb))
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-8))
            uncertainty_correlation = uncertainty_error_correlation(
                output["uncertainty"], error_map
            )
            metrics = {
                key: value for key, value in losses.items() if key != "G"
            }
            metrics.update({"psnr": psnr, "uncertainty_error_corr": uncertainty_correlation})
            batch = unstain_od.shape[0]
            for key, value in metrics.items():
                totals[key] += float(value) * batch
            count += batch
            if preview is None:
                preview = tuple(
                    value.detach().float().cpu()
                    for value in (
                        unstain_od,
                        predicted_od,
                        output["concentrations"],
                        target_concentrations,
                        output["uncertainty"],
                        output["rgb"],
                        real_rgb,
                        error_map,
                    )
                )
        result = {key: value / count for key, value in totals.items()}
        result["selection_score"] = (
            result["rgb_l1"] + (1.0 - result["ssim_score"])
            + 0.25 * (result["h_l1"] + result["e_l1"])
            + 0.25 * result["h_morphology"]
            + 0.10 * result["uncertainty_calibration"]
        )
        return result, preview

    def save_preview(self, epoch, preview):
        (
            unstain_od, predicted_od, predicted_he, target_he,
            uncertainty, fake_rgb, real_rgb, error_map,
        ) = preview
        rows = min(self.params["preview_count"], unstain_od.shape[0])
        labels = [
            "Unstain OD", "Predicted H&E OD", "Predicted H", "Predicted E",
            "Target H", "Target E", "Uncertainty", "Generated H&E",
            "Real H&E", "Absolute error",
        ]
        fig, axes = plt.subplots(rows, len(labels), figsize=(30, 3 * rows), squeeze=False)
        for row in range(rows):
            panels = [
                denormalize(unstain_od[row, 0]).numpy(),
                denormalize(predicted_od[row, 0]).numpy(),
                predicted_he[row, 0].numpy(), predicted_he[row, 1].numpy(),
                target_he[row, 0].numpy(), target_he[row, 1].numpy(),
                uncertainty[row, 0].numpy(),
                denormalize(fake_rgb[row]).permute(1, 2, 0).numpy().clip(0, 1),
                denormalize(real_rgb[row]).permute(1, 2, 0).numpy().clip(0, 1),
                error_map[row, 0].numpy(),
            ]
            for column, panel in enumerate(panels):
                if panel.ndim == 2:
                    axes[row, column].imshow(panel, cmap="magma" if column >= 2 else "gray")
                else:
                    axes[row, column].imshow(panel)
                axes[row, column].set_title(labels[column])
                axes[row, column].axis("off")
        fig.suptitle(f"Pathology-constrained staining — epoch {epoch + 1}", y=1.01)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"epoch_{epoch + 1:04d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def save_checkpoint(self, epoch, is_best):
        state = {
            "epoch": epoch,
            "G_color": self.G.state_dict(), "D_color": self.D.state_dict(),
            "opt_g": self.opt_g.state_dict(), "opt_d": self.opt_d.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(),
            "scaler_g": self.scaler_g.state_dict(), "scaler_d": self.scaler_d.state_dict(),
            "history": self.history, "best_score": self.best_score,
            "params": self.params,
        }
        torch.save(state, self.checkpoint_dir / "latest.pt")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pt")
        if (epoch + 1) % self.params["save_every"] == 0:
            torch.save(state, self.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt")

    def fit(self):
        for epoch in range(self.params["num_epochs"]):
            train_metrics = self.train_epoch(epoch)
            val_metrics, preview = self.validate()
            self.scheduler_g.step(); self.scheduler_d.step()
            row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
            self.history.append(row)
            is_best = val_metrics["selection_score"] < self.best_score
            if is_best:
                self.best_score = val_metrics["selection_score"]
            self.save_checkpoint(epoch, is_best)
            self.save_preview(epoch, preview)
            print("train:", {key: round(value, 4) for key, value in train_metrics.items()})
            print("val:", {key: round(value, 4) for key, value in val_metrics.items()})
            print("stain basis:\n", self.G.stain_basis().detach().cpu().numpy())
            if is_best:
                print(f"new best pathology colorizer: epoch {epoch + 1}")
            with (self.output_dir / "history.json").open("w") as file:
                json.dump(self.history, file, indent=2)
