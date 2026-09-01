from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import openslide
import pyvips
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from cyclegan_core import ResNetGenerator
from two_stage_virtual_staining import ODColorizer, image_to_od, prepare_physical_view


def load_two_stage_models(structure_checkpoint, color_checkpoint, device=None):
    """Load the trusted local Stage 1 and Stage 2 best checkpoints for inference."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    structure_checkpoint = Path(structure_checkpoint)
    color_checkpoint = Path(color_checkpoint)
    if not structure_checkpoint.is_file():
        raise FileNotFoundError(structure_checkpoint)
    if not color_checkpoint.is_file():
        raise FileNotFoundError(color_checkpoint)

    structure_state = torch.load(
        structure_checkpoint, map_location="cpu", weights_only=False
    )
    color_state = torch.load(color_checkpoint, map_location="cpu", weights_only=False)
    structure_params = structure_state["params"]
    color_params = color_state["params"]

    structure = ResNetGenerator(
        1, structure_params["ngf"], structure_params["residual_blocks"]
    )
    structure.load_state_dict(structure_state["G_AB"])
    colorizer = ODColorizer(color_params["base_channels"])
    colorizer.load_state_dict(color_state["G_color"])
    structure = structure.to(device).eval()
    colorizer = colorizer.to(device).eval()
    for model in (structure, colorizer):
        for parameter in model.parameters():
            parameter.requires_grad = False

    metadata = {
        "device": str(device),
        "structure_epoch": int(structure_state["epoch"]) + 1,
        "color_epoch": int(color_state["epoch"]) + 1,
        "unstain_od_max": float(structure_state["od_max"]),
        "input_size": int(structure_params["input_size"]),
        "source_mpp": float(structure_params["source_mpp"]),
        "target_mpp": float(structure_params["target_mpp"]),
    }
    return structure, colorizer, metadata


def preprocess_unstain(image, od_max):
    """Convert a PIL RGB image to the globally calibrated 1-channel OD tensor."""
    return image_to_od(image.convert("RGB"), od_max).unsqueeze(0)


@torch.inference_mode()
def infer_batch(unstain_od, structure, colorizer, device):
    unstain_od = unstain_od.to(device, non_blocking=True)
    amp_enabled = torch.device(device).type == "cuda"
    with torch.amp.autocast(torch.device(device).type, enabled=amp_enabled):
        predicted_hne_od = structure(unstain_od)
        generated_rgb = colorizer(predicted_hne_od)
    return predicted_hne_od.float().cpu(), generated_rgb.float().cpu()


@torch.inference_mode()
def infer_patch(image, structure, colorizer, metadata):
    """Infer one PIL patch and return (unstain OD, predicted H&E OD, generated RGB)."""
    params = {
        "input_size": metadata["input_size"],
        "original_size": image.size[0],
        "source_mpp": metadata["source_mpp"],
        "target_mpp": metadata["target_mpp"],
    }
    if image.size[0] != image.size[1]:
        raise ValueError(f"Patch must be square, got {image.size}")
    prepared = prepare_physical_view(image.convert("RGB"), params)
    unstain_od = preprocess_unstain(prepared, metadata["unstain_od_max"])
    predicted_od, generated_rgb = infer_batch(
        unstain_od, structure, colorizer, metadata["device"]
    )
    generated = (
        ((generated_rgb[0].clamp(-1, 1) + 1) * 127.5)
        .permute(1, 2, 0)
        .byte()
        .numpy()
    )
    return unstain_od[0], predicted_od[0], generated


def slide_mpp(slide, fallback_mpp=0.5):
    try:
        x = float(slide.properties.get("openslide.mpp-x", "nan"))
        y = float(slide.properties.get("openslide.mpp-y", "nan"))
    except (TypeError, ValueError):
        x = y = math.nan
    if not (math.isfinite(x) and math.isfinite(y) and 0.05 <= x <= 10 and 0.05 <= y <= 10):
        return float(fallback_mpp), True
    return (x + y) / 2, False


def _read_target_tile(slide, x, y, tile_size, source_mpp, target_mpp, level):
    """Read an exact target-MPP tile; coordinates x/y are in the output space."""
    downsample = float(slide.level_downsamples[level])
    level0_x = int(round(x * target_mpp / source_mpp))
    level0_y = int(round(y * target_mpp / source_mpp))
    level0_extent = tile_size * target_mpp / source_mpp
    read_size = max(1, int(math.ceil(level0_extent / downsample)))
    rgba = slide.read_region((level0_x, level0_y), level, (read_size, read_size)).convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    rgb = white.convert("RGB")
    if rgb.size != (tile_size, tile_size):
        rgb = rgb.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
    return rgb


def _blend_window(tile_size):
    axis = np.hanning(tile_size).astype(np.float32)
    # Non-zero borders are required at the four outer WSI edges.
    axis = np.maximum(axis, 0.05)
    return np.outer(axis, axis).astype(np.float32)


def _dilated_tissue_mask(gray, threshold, dilation):
    mask = torch.from_numpy(gray < threshold)[None, None].float()
    dilation = max(1, int(dilation))
    if dilation % 2 == 0:
        dilation += 1
    if dilation > 1:
        mask = F.max_pool2d(mask, dilation, stride=1, padding=dilation // 2)
    return mask[0, 0].numpy() > 0


def _tile_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def save_pyramidal_tiff(rgb, output_path, mpp, jpeg_quality=90):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, bands = rgb.shape
    if bands != 3:
        raise ValueError(f"Expected RGB array, got {rgb.shape}")
    image = pyvips.Image.new_from_memory(rgb.tobytes(), width, height, bands, "uchar")
    temporary = output_path.with_name(f".{output_path.stem}.tmp.tiff")
    if temporary.exists():
        temporary.unlink()
    pixels_per_mm = 1000.0 / float(mpp)
    image.tiffsave(
        str(temporary),
        tile=True,
        tile_width=256,
        tile_height=256,
        pyramid=True,
        compression="jpeg",
        Q=int(jpeg_quality),
        bigtiff=True,
        xres=pixels_per_mm,
        yres=pixels_per_mm,
        resunit="cm",
    )
    temporary.replace(output_path)


def infer_wsi(
    slide_path,
    output_path,
    structure,
    colorizer,
    metadata,
    *,
    tile_size=512,
    overlap=64,
    batch_size=4,
    fallback_mpp=0.5,
    tissue_gray_threshold=0.98,
    minimum_tissue_fraction=0.002,
    white_background_threshold=0.995,
    background_dilation=31,
    jpeg_quality=90,
):
    """Run overlap-tiled inference and save an OpenSlide-readable pyramidal TIFF."""
    slide_path, output_path = Path(slide_path), Path(output_path)
    if not slide_path.is_file():
        raise FileNotFoundError(slide_path)
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    stride = tile_size - overlap
    target_mpp = float(metadata["target_mpp"])
    device = torch.device(metadata["device"])

    with openslide.OpenSlide(str(slide_path)) as slide:
        source_mpp, used_fallback = slide_mpp(slide, fallback_mpp)
        output_width = int(math.ceil(slide.dimensions[0] * source_mpp / target_mpp))
        output_height = int(math.ceil(slide.dimensions[1] * source_mpp / target_mpp))
        desired_downsample = target_mpp / source_mpp
        level = slide.get_best_level_for_downsample(desired_downsample)
        xs = _tile_starts(output_width, tile_size, stride)
        ys = _tile_starts(output_height, tile_size, stride)
        positions = [(x, y) for y in ys for x in xs]
        window = _blend_window(tile_size)

        with tempfile.TemporaryDirectory(prefix="two_stage_wsi_") as temporary_dir:
            temporary_dir = Path(temporary_dir)
            accumulation = np.memmap(
                temporary_dir / "rgb.float32", mode="w+", dtype=np.float32,
                shape=(output_height, output_width, 3),
            )
            weights = np.memmap(
                temporary_dir / "weight.float32", mode="w+", dtype=np.float32,
                shape=(output_height, output_width),
            )
            accumulation[:] = 0
            weights[:] = 0

            pending_images, pending_positions, pending_gray, pending_tissue = [], [], [], []

            def flush_batch():
                if not pending_images:
                    return
                od_batch = torch.cat(
                    [preprocess_unstain(image, metadata["unstain_od_max"]) for image in pending_images],
                    dim=0,
                )
                _, generated = infer_batch(od_batch, structure, colorizer, device)
                generated = (
                    ((generated.clamp(-1, 1) + 1) * 127.5)
                    .permute(0, 2, 3, 1).byte().numpy()
                )
                for rgb, gray, tissue, (x, y) in zip(
                    generated, pending_gray, pending_tissue, pending_positions
                ):
                    rgb[~tissue] = 255
                    rgb[gray >= white_background_threshold] = 255
                    valid_w = min(tile_size, output_width - x)
                    valid_h = min(tile_size, output_height - y)
                    local_window = window[:valid_h, :valid_w]
                    accumulation[y:y + valid_h, x:x + valid_w] += (
                        rgb[:valid_h, :valid_w].astype(np.float32) * local_window[..., None]
                    )
                    weights[y:y + valid_h, x:x + valid_w] += local_window
                pending_images.clear(); pending_positions.clear()
                pending_gray.clear(); pending_tissue.clear()

            for x, y in tqdm(positions, desc=f"WSI inference at {target_mpp:g} MPP"):
                image = _read_target_tile(
                    slide, x, y, tile_size, source_mpp, target_mpp, level
                )
                gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
                tissue_fraction = float((gray < tissue_gray_threshold).mean())
                if tissue_fraction < minimum_tissue_fraction:
                    valid_w = min(tile_size, output_width - x)
                    valid_h = min(tile_size, output_height - y)
                    local_window = window[:valid_h, :valid_w]
                    accumulation[y:y + valid_h, x:x + valid_w] += 255 * local_window[..., None]
                    weights[y:y + valid_h, x:x + valid_w] += local_window
                    continue
                pending_images.append(image)
                pending_positions.append((x, y))
                pending_gray.append(gray)
                pending_tissue.append(
                    _dilated_tissue_mask(gray, tissue_gray_threshold, background_dilation)
                )
                if len(pending_images) >= batch_size:
                    flush_batch()
            flush_batch()

            output = np.empty((output_height, output_width, 3), dtype=np.uint8)
            for row in range(0, output_height, 512):
                end = min(output_height, row + 512)
                denominator = np.maximum(weights[row:end], 1e-6)[..., None]
                output[row:end] = np.clip(accumulation[row:end] / denominator, 0, 255).astype(np.uint8)

    save_pyramidal_tiff(output, output_path, target_mpp, jpeg_quality)
    with openslide.OpenSlide(str(output_path)) as generated_slide:
        saved_mpp = float(generated_slide.properties.get("openslide.mpp-x", "nan"))
        saved_dimensions = generated_slide.dimensions
    return {
        "input": str(slide_path),
        "output": str(output_path),
        "source_mpp": source_mpp,
        "used_fallback_mpp": used_fallback,
        "target_mpp": target_mpp,
        "saved_mpp": saved_mpp,
        "output_dimensions": saved_dimensions,
        "read_level": level,
        "tiles": len(positions),
    }
