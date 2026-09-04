from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import openslide
import pandas as pd
from PIL import Image
from scipy.stats import wilcoxon
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from two_stage_inference import save_pyramidal_tiff, slide_mpp


HE_BASIS = np.asarray(
    [[0.650, 0.072], [0.704, 0.990], [0.286, 0.105]], dtype=np.float32
)
HE_BASIS /= np.linalg.norm(HE_BASIS, axis=0, keepdims=True)
HE_INVERSE = np.linalg.pinv(HE_BASIS)

METRIC_DIRECTIONS = {
    "ssim_full_rgb": "higher",
    "ssim_full_gray": "higher",
    "ssim_tissue_rgb": "higher",
    "ssim_coarse_rgb": "higher",
    "psnr_full_db": "higher",
    "psnr_tissue_db": "higher",
    "mae_full": "lower",
    "mae_tissue": "lower",
    "rmse_tissue": "lower",
    "delta_e2000_tissue": "lower",
    "gradient_corr_tissue": "higher",
    "h_concentration_mae": "lower",
    "e_concentration_mae": "lower",
    "h_spatial_corr": "higher",
    "laplacian_energy_log_error": "lower",
    "tissue_dice": "higher",
    "lpips": "lower",
}

PRIMARY_METRICS = (
    "ssim_full_rgb",
    "ssim_tissue_rgb",
    "ssim_coarse_rgb",
    "psnr_full_db",
    "lpips",
    "delta_e2000_tissue",
    "gradient_corr_tissue",
    "h_spatial_corr",
    "tissue_dice",
)


def _rgba_to_white_rgb(image):
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    return white.convert("RGB")


def read_slide_at_mpp(slide_path, target_mpp, fallback_mpp=0.5):
    """Read a WSI at an exact physical pixel size and return RGB plus metadata."""
    slide_path = Path(slide_path)
    if not slide_path.is_file():
        raise FileNotFoundError(slide_path)
    with openslide.OpenSlide(str(slide_path)) as slide:
        source_mpp, used_fallback = slide_mpp(slide, fallback_mpp)
        target_width = int(math.ceil(slide.dimensions[0] * source_mpp / target_mpp))
        target_height = int(math.ceil(slide.dimensions[1] * source_mpp / target_mpp))
        desired_downsample = target_mpp / source_mpp
        level = slide.get_best_level_for_downsample(desired_downsample)
        image = _rgba_to_white_rgb(
            slide.read_region((0, 0), level, slide.level_dimensions[level])
        )
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
        metadata = {
            "path": str(slide_path),
            "source_mpp": float(source_mpp),
            "target_mpp": float(target_mpp),
            "used_fallback_mpp": bool(used_fallback),
            "level": int(level),
            "level0_dimensions": tuple(int(value) for value in slide.dimensions),
            "target_dimensions": (target_width, target_height),
        }
    return np.asarray(image, dtype=np.uint8), metadata


def discover_heldout_wsi(
    patch_data_dir,
    unstain_wsi_dir,
    hne_wsi_dir,
    registration_dir,
    *,
    image_ext="png",
    val_fraction=0.1,
    seed=42,
):
    """Reproduce the training split and resolve every held-out WSI triplet."""
    from cyclegan_core import slide_key
    from two_stage_virtual_staining import collect_matched_pairs, split_pairs_by_slide

    pairs = collect_matched_pairs(patch_data_dir, image_ext)
    _, validation_pairs = split_pairs_by_slide(pairs, val_fraction, seed)
    heldout_keys = sorted({slide_key(pair[0]) for pair in validation_pairs})
    rows = []
    for slide in heldout_keys:
        unstain_path = Path(unstain_wsi_dir) / f"{slide}.tiff"
        hne_path = Path(hne_wsi_dir) / f"{slide}.tiff"
        registration_path = Path(registration_dir) / f"{slide}_registration.json"
        missing = [
            str(path) for path in (unstain_path, hne_path, registration_path)
            if not path.is_file()
        ]
        rows.append({
            "case_id": case_id_from_slide(slide),
            "slide": slide,
            "unstain_path": str(unstain_path),
            "hne_path": str(hne_path),
            "registration_path": str(registration_path),
            "ready": not missing,
            "missing": " | ".join(missing),
        })
    return pd.DataFrame(rows)


def load_registration(registration_json):
    registration_json = Path(registration_json)
    if not registration_json.is_file():
        raise FileNotFoundError(registration_json)
    with registration_json.open(encoding="utf-8") as file:
        return json.load(file)


def target_mpp_affine(fullres_moving_to_fixed, fixed_mpp, moving_mpp, target_mpp):
    """Convert a level-0 moving->fixed affine matrix to a target-MPP raster."""
    matrix = np.asarray(fullres_moving_to_fixed, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 transform, got {matrix.shape}")
    fixed_scale = float(fixed_mpp) / float(target_mpp)
    moving_scale = float(moving_mpp) / float(target_mpp)
    fixed_scaling = np.diag([fixed_scale, fixed_scale, 1.0])
    inverse_moving_scaling = np.diag([1.0 / moving_scale, 1.0 / moving_scale, 1.0])
    return fixed_scaling @ matrix @ inverse_moving_scaling


def register_real_hne_to_unstain(
    unstain_wsi,
    hne_wsi,
    registration_json,
    target_mpp,
    output_path=None,
    fallback_mpp=0.5,
):
    """Warp native real H&E pixels into the fixed Unstain WSI coordinate system."""
    fixed_rgb, fixed_info = read_slide_at_mpp(
        unstain_wsi, target_mpp, fallback_mpp
    )
    moving_rgb, moving_info = read_slide_at_mpp(
        hne_wsi, target_mpp, fallback_mpp
    )
    registration = load_registration(registration_json)
    matrix_fullres = registration["transforms"][
        "moving_hne_to_fixed_unstain_fullres_3x3"
    ]
    matrix_target = target_mpp_affine(
        matrix_fullres,
        fixed_info["source_mpp"],
        moving_info["source_mpp"],
        target_mpp,
    )
    height, width = fixed_rgb.shape[:2]
    registered = cv2.warpAffine(
        moving_rgb,
        matrix_target[:2].astype(np.float32),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    moving_valid = np.ones(moving_rgb.shape[:2], dtype=np.uint8)
    valid_mask = cv2.warpAffine(
        moving_valid,
        matrix_target[:2].astype(np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    if output_path is not None:
        save_pyramidal_tiff(registered, output_path, target_mpp, jpeg_quality=95)
    info = {
        "fixed": fixed_info,
        "moving": moving_info,
        "registration_json": str(registration_json),
        "registration_quality": registration.get("quality", {}),
        "target_mpp_affine": matrix_target.tolist(),
        "valid_fraction": float(valid_mask.mean()),
    }
    return fixed_rgb, moving_rgb, registered, valid_mask, info


def tissue_mask_hne(rgb):
    rgb_float = rgb.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mask = ((hsv[..., 1] > 0.05) & (gray < 0.98)) | (gray < 0.82)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask.astype(bool)


def _masked_ssim(candidate, reference, mask, window_size=11):
    candidate = candidate.astype(np.float32) / 255.0
    reference = reference.astype(np.float32) / 255.0
    c1, c2 = 0.01**2, 0.03**2
    channel_scores = []
    for channel in range(3):
        x, y = candidate[..., channel], reference[..., channel]
        mu_x = cv2.boxFilter(
            x, -1, (window_size, window_size), normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        mu_y = cv2.boxFilter(
            y, -1, (window_size, window_size), normalize=True,
            borderType=cv2.BORDER_REFLECT,
        )
        sigma_x = cv2.boxFilter(x * x, -1, (window_size, window_size), normalize=True) - mu_x**2
        sigma_y = cv2.boxFilter(y * y, -1, (window_size, window_size), normalize=True) - mu_y**2
        sigma_xy = cv2.boxFilter(x * y, -1, (window_size, window_size), normalize=True) - mu_x * mu_y
        score_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2) + 1e-12
        )
        channel_scores.append(float(score_map[mask].mean()))
    return float(np.mean(channel_scores))


def _full_ssim(candidate, reference, grayscale=False):
    if grayscale:
        candidate = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
        reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
        return float(structural_similarity(candidate, reference, data_range=255))
    return float(
        structural_similarity(
            candidate, reference, data_range=255, channel_axis=2
        )
    )


def _coarse_ssim(candidate, reference, downsample=4):
    height, width = candidate.shape[:2]
    output_size = (
        max(16, width // int(downsample)),
        max(16, height // int(downsample)),
    )
    candidate_small = cv2.resize(candidate, output_size, interpolation=cv2.INTER_AREA)
    reference_small = cv2.resize(reference, output_size, interpolation=cv2.INTER_AREA)
    return _full_ssim(candidate_small, reference_small)


def _stain_concentrations(rgb):
    rgb_float = np.clip(rgb.astype(np.float32) / 255.0, 1 / 255, 1.0)
    optical_density = -np.log(rgb_float)
    concentrations = np.einsum("kc,hwc->hwk", HE_INVERSE, optical_density)
    return np.clip(concentrations, 0.0, 4.0)


def _gradient_correlation(candidate, reference, mask):
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    def magnitude(gray):
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(dx * dx + dy * dy)

    x, y = magnitude(candidate_gray)[mask], magnitude(reference_gray)[mask]
    if x.size < 2 or x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_correlation(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _masked_block_means(values, mask, grid_size=16):
    values = np.asarray(values, dtype=np.float32)
    mask_float = np.asarray(mask, dtype=np.float32)
    size = (int(grid_size), int(grid_size))
    weighted = cv2.resize(values * mask_float, size, interpolation=cv2.INTER_AREA)
    weights = cv2.resize(mask_float, size, interpolation=cv2.INTER_AREA)
    valid = weights > 0.05
    means = np.zeros_like(weighted)
    means[valid] = weighted[valid] / weights[valid]
    return means[valid]


def _laplacian_energy(gray, mask):
    laplacian = cv2.Laplacian(gray.astype(np.float32) / 255.0, cv2.CV_32F)
    return float(np.mean(laplacian[mask] ** 2))


def calculate_masked_metrics(candidate, reference, mask):
    if mask.sum() < 64:
        raise ValueError("Evaluation mask contains fewer than 64 pixels")
    candidate_float = candidate.astype(np.float32) / 255.0
    reference_float = reference.astype(np.float32) / 255.0
    difference = candidate_float - reference_float
    selected = difference[mask]
    full_mae = float(np.abs(difference).mean())
    full_mse = float((difference**2).mean())
    tissue_mae = float(np.abs(selected).mean())
    tissue_mse = float((selected**2).mean())
    tissue_rmse = math.sqrt(tissue_mse)
    full_psnr = -10.0 * math.log10(max(full_mse, 1e-12))
    tissue_psnr = -10.0 * math.log10(max(tissue_mse, 1e-12))

    candidate_lab = rgb2lab(candidate_float)
    reference_lab = rgb2lab(reference_float)
    delta_e = deltaE_ciede2000(candidate_lab, reference_lab)
    candidate_he = _stain_concentrations(candidate)
    reference_he = _stain_concentrations(reference)
    candidate_tissue = tissue_mask_hne(candidate)
    reference_tissue = tissue_mask_hne(reference)
    intersection = np.logical_and(candidate_tissue, reference_tissue).sum()
    tissue_dice = (2.0 * intersection + 1e-6) / (
        candidate_tissue.sum() + reference_tissue.sum() + 1e-6
    )
    candidate_h_grid = _masked_block_means(candidate_he[..., 0], mask)
    reference_h_grid = _masked_block_means(reference_he[..., 0], mask)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    candidate_laplacian = _laplacian_energy(candidate_gray, mask)
    reference_laplacian = _laplacian_energy(reference_gray, mask)
    laplacian_error = abs(
        math.log((candidate_laplacian + 1e-12) / (reference_laplacian + 1e-12))
    )
    return {
        "ssim_full_rgb": _full_ssim(candidate, reference),
        "ssim_full_gray": _full_ssim(candidate, reference, grayscale=True),
        "ssim_tissue_rgb": _masked_ssim(candidate, reference, mask),
        "ssim_coarse_rgb": _coarse_ssim(candidate, reference, downsample=4),
        "psnr_full_db": full_psnr,
        "psnr_tissue_db": tissue_psnr,
        "mae_full": full_mae,
        "mae_tissue": tissue_mae,
        "rmse_tissue": tissue_rmse,
        "delta_e2000_tissue": float(delta_e[mask].mean()),
        "gradient_corr_tissue": _gradient_correlation(candidate, reference, mask),
        "h_concentration_mae": float(
            np.abs(candidate_he[..., 0] - reference_he[..., 0])[mask].mean()
        ),
        "e_concentration_mae": float(
            np.abs(candidate_he[..., 1] - reference_he[..., 1])[mask].mean()
        ),
        "h_spatial_corr": _safe_correlation(candidate_h_grid, reference_h_grid),
        "laplacian_energy_log_error": float(laplacian_error),
        "tissue_dice": float(tissue_dice),
    }


class DeepFeatureEvaluator:
    """Compute paired LPIPS and standard pytorch-fid Inception features."""

    def __init__(self, device=None, batch_size=8):
        import lpips
        import torch
        from pytorch_fid.inception import InceptionV3

        self.torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = int(batch_size)
        self.lpips_model = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3([block]).to(self.device).eval()
        for model in (self.lpips_model, self.inception):
            for parameter in model.parameters():
                parameter.requires_grad = False

    def evaluate_pairs(self, candidates, references):
        torch = self.torch
        if len(candidates) != len(references) or not candidates:
            raise ValueError("Deep metrics require equally sized non-empty image lists")
        lpips_values, candidate_features, reference_features = [], [], []
        for start in range(0, len(candidates), self.batch_size):
            stop = start + self.batch_size
            candidate = torch.from_numpy(
                np.stack(candidates[start:stop]).copy()
            ).permute(0, 3, 1, 2).float().div_(255).to(self.device)
            reference = torch.from_numpy(
                np.stack(references[start:stop]).copy()
            ).permute(0, 3, 1, 2).float().div_(255).to(self.device)
            with torch.inference_mode():
                lpips_batch = self.lpips_model(candidate * 2 - 1, reference * 2 - 1)
                candidate_feature = self.inception(candidate)[0]
                reference_feature = self.inception(reference)[0]
                if candidate_feature.shape[-2:] != (1, 1):
                    candidate_feature = torch.nn.functional.adaptive_avg_pool2d(
                        candidate_feature, output_size=(1, 1)
                    )
                    reference_feature = torch.nn.functional.adaptive_avg_pool2d(
                        reference_feature, output_size=(1, 1)
                    )
            lpips_values.append(lpips_batch.flatten().cpu().numpy())
            candidate_features.append(candidate_feature.flatten(1).cpu().numpy())
            reference_features.append(reference_feature.flatten(1).cpu().numpy())
        return {
            "lpips": np.concatenate(lpips_values).astype(np.float64),
            "candidate_features": np.concatenate(candidate_features).astype(np.float64),
            "reference_features": np.concatenate(reference_features).astype(np.float64),
        }


def _polynomial_mmd(features_a, features_b):
    features_a = np.asarray(features_a, dtype=np.float64)
    features_b = np.asarray(features_b, dtype=np.float64)
    dimension = features_a.shape[1]
    kernel_aa = (features_a @ features_a.T / dimension + 1.0) ** 3
    kernel_bb = (features_b @ features_b.T / dimension + 1.0) ** 3
    kernel_ab = (features_a @ features_b.T / dimension + 1.0) ** 3
    count_a, count_b = len(features_a), len(features_b)
    within_a = (kernel_aa.sum() - np.trace(kernel_aa)) / (count_a * (count_a - 1))
    within_b = (kernel_bb.sum() - np.trace(kernel_bb)) / (count_b * (count_b - 1))
    return float(within_a + within_b - 2.0 * kernel_ab.mean())


def summarize_distribution_features(feature_payloads, kid_subsets=100, seed=42):
    """Return FID/KID point estimates; these are distribution-, not pair-metrics."""
    from pytorch_fid.fid_score import calculate_frechet_distance

    generator = np.random.default_rng(seed)
    rows = []
    for comparison, payload in feature_payloads.items():
        candidate = np.concatenate(payload["candidate_features"], axis=0)
        reference = np.concatenate(payload["reference_features"], axis=0)
        if min(len(candidate), len(reference)) < 2:
            continue
        fid = calculate_frechet_distance(
            candidate.mean(axis=0), np.cov(candidate, rowvar=False),
            reference.mean(axis=0), np.cov(reference, rowvar=False),
        )
        subset_size = min(100, len(candidate), len(reference))
        kid_values = []
        for _ in range(int(kid_subsets)):
            candidate_index = generator.choice(len(candidate), subset_size, replace=False)
            reference_index = generator.choice(len(reference), subset_size, replace=False)
            kid_values.append(
                _polynomial_mmd(candidate[candidate_index], reference[reference_index])
            )
        rows.append({
            "comparison": comparison,
            "n_candidate_tiles": len(candidate),
            "n_reference_tiles": len(reference),
            "fid_2048": float(fid),
            "kid_mean": float(np.mean(kid_values)),
            "kid_std": float(np.std(kid_values, ddof=1)) if len(kid_values) > 1 else 0.0,
            "kid_subset_size": subset_size,
        })
    return pd.DataFrame(rows)


def evaluate_wsi_tiles(
    unstain_rgb,
    generated_rgb,
    registered_hne_rgb,
    valid_mask,
    *,
    slide_name,
    tile_size=512,
    stride=512,
    minimum_tissue_fraction=0.10,
    deep_evaluator=None,
):
    arrays = [unstain_rgb, generated_rgb, registered_hne_rgb]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(f"WSI raster shapes do not match: {[array.shape for array in arrays]}")
    height, width = registered_hne_rgb.shape[:2]
    real_tissue = tissue_mask_hne(registered_hne_rgb) & valid_mask
    records = []
    deep_pairs = {}
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            region = np.s_[y:y + tile_size, x:x + tile_size]
            local_valid = valid_mask[region]
            local_mask = real_tissue[region]
            tissue_fraction = float(local_mask.mean())
            if tissue_fraction < minimum_tissue_fraction or local_valid.mean() < 0.95:
                continue
            reference = registered_hne_rgb[region]
            for comparison, candidate in (
                ("Unstain baseline", unstain_rgb[region]),
                ("Virtual H&E", generated_rgb[region]),
            ):
                record = {
                    "slide": slide_name,
                    "tile_x": x,
                    "tile_y": y,
                    "comparison": comparison,
                    "tissue_fraction": tissue_fraction,
                    "evaluated_pixels": int(local_mask.sum()),
                }
                record.update(calculate_masked_metrics(candidate, reference, local_mask))
                records.append(record)
                if deep_evaluator is not None:
                    payload = deep_pairs.setdefault(
                        comparison,
                        {"row_indices": [], "candidates": [], "references": []},
                    )
                    payload["row_indices"].append(len(records) - 1)
                    payload["candidates"].append(candidate)
                    payload["references"].append(reference)
    if not records:
        raise RuntimeError(
            "No evaluation tiles passed the valid/tissue thresholds; lower "
            "minimum_tissue_fraction or verify registration."
        )
    frame = pd.DataFrame.from_records(records)
    feature_payload = {}
    if deep_evaluator is not None:
        for comparison, payload in deep_pairs.items():
            result = deep_evaluator.evaluate_pairs(
                payload["candidates"], payload["references"]
            )
            frame.loc[payload["row_indices"], "lpips"] = result["lpips"]
            feature_payload[comparison] = {
                "candidate_features": result["candidate_features"],
                "reference_features": result["reference_features"],
            }
    return frame, real_tissue, feature_payload


def _bootstrap_mean_ci(values, iterations=2000, seed=42):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _fdr_bh(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def summarize_wsi_metrics(tile_metrics, bootstrap_iterations=2000, seed=42):
    metric_names = [
        metric for metric in METRIC_DIRECTIONS if metric in tile_metrics.columns
    ]
    summary_rows = []
    for comparison, group in tile_metrics.groupby("comparison", sort=False):
        for metric in metric_names:
            values = group[metric].to_numpy(dtype=np.float64)
            ci_low, ci_high = _bootstrap_mean_ci(
                values, bootstrap_iterations, seed
            )
            summary_rows.append({
                "comparison": comparison,
                "metric": metric,
                "direction": METRIC_DIRECTIONS[metric],
                "n_tiles": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            })
    summary = pd.DataFrame(summary_rows)

    keys = ["slide", "tile_x", "tile_y"]
    baseline = tile_metrics[tile_metrics["comparison"] == "Unstain baseline"].set_index(keys)
    generated = tile_metrics[tile_metrics["comparison"] == "Virtual H&E"].set_index(keys)
    common = baseline.index.intersection(generated.index)
    improvement_rows = []
    for metric in metric_names:
        baseline_values = baseline.loc[common, metric].to_numpy(dtype=np.float64)
        generated_values = generated.loc[common, metric].to_numpy(dtype=np.float64)
        sign = 1.0 if METRIC_DIRECTIONS[metric] == "higher" else -1.0
        improvement = sign * (generated_values - baseline_values)
        ci_low, ci_high = _bootstrap_mean_ci(
            improvement, bootstrap_iterations, seed
        )
        try:
            p_value = float(wilcoxon(improvement, alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        improvement_rows.append({
            "metric": metric,
            "direction": METRIC_DIRECTIONS[metric],
            "n_paired_tiles": len(common),
            "unstain_mean": float(baseline_values.mean()),
            "virtual_hne_mean": float(generated_values.mean()),
            "mean_improvement": float(improvement.mean()),
            "improvement_ci95_low": ci_low,
            "improvement_ci95_high": ci_high,
            "wilcoxon_p": p_value,
        })
    improvement = pd.DataFrame(improvement_rows)
    improvement["fdr_bh_q"] = _fdr_bh(improvement["wilcoxon_p"].to_numpy())
    return summary, improvement


def case_id_from_slide(slide_name):
    return str(slide_name).split("(", 1)[0]


def summarize_cohort_metrics(tile_metrics, bootstrap_iterations=5000, seed=42):
    """Aggregate tiles equally by slide and then by independent case."""
    metric_names = [
        metric for metric in METRIC_DIRECTIONS if metric in tile_metrics.columns
    ]
    grouped = tile_metrics.copy()
    grouped["case_id"] = grouped["slide"].map(case_id_from_slide)
    slide_metrics = (
        grouped.groupby(["case_id", "slide", "comparison"], as_index=False)
        .agg(n_tiles=("tile_x", "size"), **{
            metric: (metric, "mean") for metric in metric_names
        })
    )
    case_metrics = (
        slide_metrics.groupby(["case_id", "comparison"], as_index=False)
        .agg(n_slides=("slide", "nunique"), n_tiles=("n_tiles", "sum"), **{
            metric: (metric, "mean") for metric in metric_names
        })
    )

    summary_rows = []
    for comparison, group in case_metrics.groupby("comparison", sort=False):
        for metric in metric_names:
            values = group[metric].to_numpy(dtype=np.float64)
            ci_low, ci_high = _bootstrap_mean_ci(
                values, bootstrap_iterations, seed
            )
            summary_rows.append({
                "comparison": comparison,
                "metric": metric,
                "direction": METRIC_DIRECTIONS[metric],
                "n_cases": group["case_id"].nunique(),
                "n_slides": slide_metrics.loc[
                    slide_metrics["comparison"].eq(comparison), "slide"
                ].nunique(),
                "n_tiles": int(group["n_tiles"].sum()),
                "case_mean": float(values.mean()),
                "case_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "case_median": float(np.median(values)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            })
    summary = pd.DataFrame(summary_rows)

    keys = ["case_id"]
    baseline = case_metrics[case_metrics["comparison"].eq("Unstain baseline")].set_index(keys)
    generated = case_metrics[case_metrics["comparison"].eq("Virtual H&E")].set_index(keys)
    common = baseline.index.intersection(generated.index)
    improvement_rows = []
    for metric in metric_names:
        baseline_values = baseline.loc[common, metric].to_numpy(dtype=np.float64)
        generated_values = generated.loc[common, metric].to_numpy(dtype=np.float64)
        sign = 1.0 if METRIC_DIRECTIONS[metric] == "higher" else -1.0
        improvement = sign * (generated_values - baseline_values)
        ci_low, ci_high = _bootstrap_mean_ci(
            improvement, bootstrap_iterations, seed
        )
        try:
            p_value = float(wilcoxon(improvement, alternative="two-sided").pvalue)
        except ValueError:
            p_value = 1.0
        improvement_rows.append({
            "metric": metric,
            "direction": METRIC_DIRECTIONS[metric],
            "n_paired_cases": len(common),
            "unstain_case_mean": float(baseline_values.mean()),
            "virtual_hne_case_mean": float(generated_values.mean()),
            "mean_improvement": float(improvement.mean()),
            "improvement_ci95_low": ci_low,
            "improvement_ci95_high": ci_high,
            "wilcoxon_p": p_value,
        })
    improvement = pd.DataFrame(improvement_rows)
    improvement["fdr_bh_q"] = _fdr_bh(improvement["wilcoxon_p"].to_numpy())
    return slide_metrics, case_metrics, summary, improvement
