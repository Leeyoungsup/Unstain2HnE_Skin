"""Extract pixel-aligned H&E/Unstain patch pairs from registration JSON files."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import openslide
from PIL import Image
from tqdm.auto import tqdm

from wsi_registration import create_tissue_mask, transform_points


def load_registration(json_path):
    with Path(json_path).open(encoding="utf-8") as input_file:
        return json.load(input_file)


def effective_mpp(
    slide,
    fallback_mpp=0.5,
    minimum=0.05,
    maximum=10.0,
    trust_metadata=False,
):
    """Return explicit source MPP unless trustworthy slide metadata is requested."""
    if not trust_metadata:
        return float(fallback_mpp), True
    try:
        mpp = float(slide.properties.get("openslide.mpp-x", "nan"))
    except (TypeError, ValueError):
        mpp = np.nan
    if not np.isfinite(mpp) or not minimum <= mpp <= maximum:
        return float(fallback_mpp), True
    return mpp, False


def _thumbnail(slide, max_size):
    return np.asarray(slide.get_thumbnail((max_size, max_size)).convert("RGB"))


def _center_pad(image, target_height, target_width, value):
    height, width = image.shape[:2]
    top = (target_height - height) // 2
    left = (target_width - width) // 2
    if image.ndim == 3:
        canvas = np.full(
            (target_height, target_width, image.shape[2]), value, dtype=image.dtype
        )
    else:
        canvas = np.full((target_height, target_width), value, dtype=image.dtype)
    canvas[top : top + height, left : left + width] = image
    return canvas, (left, top)


def build_common_tissue_mask(params):
    """Recreate the common tissue mask in the fixed Unstain thumbnail space."""
    moving_path = params["files"]["moving_hne"]
    fixed_path = params["files"]["fixed_unstain"]
    canvas = params["dimensions"]["registration_canvas"]
    canvas_width = int(canvas["width"])
    canvas_height = int(canvas["height"])
    max_size = max(canvas_width, canvas_height)

    with openslide.OpenSlide(moving_path) as moving_slide:
        moving_raw = _thumbnail(moving_slide, max_size)
    with openslide.OpenSlide(fixed_path) as fixed_slide:
        fixed_raw = _thumbnail(fixed_slide, max_size)

    moving, _ = _center_pad(moving_raw, canvas_height, canvas_width, 255)
    fixed, fixed_offset = _center_pad(fixed_raw, canvas_height, canvas_width, 255)
    moving_mask = create_tissue_mask(moving)
    fixed_mask = create_tissue_mask(
        fixed,
        threshold_offset=params["mask"]["fixed_unstain_threshold_offset"],
    )
    warp = np.asarray(
        params["transforms"]["fixed_to_moving_thumbnail_inverse_warp_2x3"],
        dtype=np.float32,
    )
    aligned_moving_mask = cv2.warpAffine(
        moving_mask,
        warp,
        (canvas_width, canvas_height),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    common_mask = cv2.bitwise_and(aligned_moving_mask, fixed_mask)
    return common_mask, fixed_raw.shape, fixed_offset


def _grid_starts(length, patch_size, stride):
    if patch_size > length:
        return []
    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _rectangle_sum(integral, x1, y1, x2, y2):
    return (
        integral[y2, x2]
        - integral[y1, x2]
        - integral[y2, x1]
        + integral[y1, x1]
    )


def get_valid_patch_positions(
    common_mask,
    fixed_dimensions,
    fixed_thumbnail_shape,
    fixed_thumbnail_offset,
    patch_size_wsi,
    overlap=0.2,
    tissue_threshold=0.3,
):
    """Return fixed-WSI patch top-left positions passing common-mask coverage."""
    fixed_width, fixed_height = fixed_dimensions
    thumbnail_height, thumbnail_width = fixed_thumbnail_shape[:2]
    offset_x, offset_y = fixed_thumbnail_offset
    scale_x = thumbnail_width / fixed_width
    scale_y = thumbnail_height / fixed_height
    stride = max(1, int(round(patch_size_wsi * (1.0 - overlap))))
    integral = cv2.integral((common_mask > 0).astype(np.uint8))

    positions = []
    for y in _grid_starts(fixed_height, patch_size_wsi, stride):
        for x in _grid_starts(fixed_width, patch_size_wsi, stride):
            x1 = int(np.floor(x * scale_x + offset_x))
            y1 = int(np.floor(y * scale_y + offset_y))
            x2 = int(np.ceil((x + patch_size_wsi) * scale_x + offset_x))
            y2 = int(np.ceil((y + patch_size_wsi) * scale_y + offset_y))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(common_mask.shape[1], x2), min(common_mask.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            foreground = _rectangle_sum(integral, x1, y1, x2, y2)
            tissue_ratio = float(foreground / ((x2 - x1) * (y2 - y1)))
            if tissue_ratio >= tissue_threshold:
                positions.append((x, y, tissue_ratio))
    return positions, stride


def _extract_fixed_patch(slide, x, y, patch_size_wsi, output_size):
    patch = np.asarray(
        slide.read_region(
            (int(x), int(y)), 0, (int(patch_size_wsi), int(patch_size_wsi))
        ).convert("RGB")
    )
    if patch_size_wsi != output_size:
        patch = cv2.resize(
            patch, (output_size, output_size), interpolation=cv2.INTER_AREA
        )
    return patch


def _extract_aligned_moving_patch(
    moving_slide,
    fixed_x,
    fixed_y,
    patch_size_wsi,
    output_size,
    fixed_to_moving_matrix,
):
    fixed_corners = np.array(
        [
            [fixed_x, fixed_y],
            [fixed_x + patch_size_wsi, fixed_y],
            [fixed_x + patch_size_wsi, fixed_y + patch_size_wsi],
            [fixed_x, fixed_y + patch_size_wsi],
        ],
        dtype=np.float64,
    )
    moving_corners = transform_points(fixed_corners, fixed_to_moving_matrix)
    margin = 3
    source_x = int(np.floor(moving_corners[:, 0].min())) - margin
    source_y = int(np.floor(moving_corners[:, 1].min())) - margin
    source_x2 = int(np.ceil(moving_corners[:, 0].max())) + margin
    source_y2 = int(np.ceil(moving_corners[:, 1].max())) + margin
    moving_width, moving_height = moving_slide.dimensions
    if (
        source_x < 0
        or source_y < 0
        or source_x2 > moving_width
        or source_y2 > moving_height
    ):
        return None, moving_corners

    source = np.asarray(
        moving_slide.read_region(
            (source_x, source_y), 0, (source_x2 - source_x, source_y2 - source_y)
        ).convert("RGB")
    )
    output_to_fixed = np.array(
        [
            [patch_size_wsi / output_size, 0.0, fixed_x],
            [0.0, patch_size_wsi / output_size, fixed_y],
            [0.0, 0.0, 1.0],
        ]
    )
    moving_to_local = np.array(
        [[1.0, 0.0, -source_x], [0.0, 1.0, -source_y], [0.0, 0.0, 1.0]]
    )
    output_to_source = moving_to_local @ fixed_to_moving_matrix @ output_to_fixed
    aligned = cv2.warpAffine(
        source,
        output_to_source[:2].astype(np.float32),
        (output_size, output_size),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return aligned, moving_corners


def prepare_patch_positions(
    registration_json,
    target_mpp=0.5,
    patch_size=2048,
    overlap=0.2,
    tissue_threshold=0.3,
    fallback_mpp=0.5,
    trust_slide_mpp=False,
):
    params = load_registration(registration_json)
    common_mask, fixed_thumbnail_shape, fixed_thumbnail_offset = (
        build_common_tissue_mask(params)
    )
    with openslide.OpenSlide(params["files"]["fixed_unstain"]) as fixed_slide:
        fixed_mpp, used_fallback = effective_mpp(
            fixed_slide, fallback_mpp, trust_metadata=trust_slide_mpp
        )
        patch_size_wsi = max(1, int(round(patch_size * target_mpp / fixed_mpp)))
        positions, stride = get_valid_patch_positions(
            common_mask=common_mask,
            fixed_dimensions=fixed_slide.dimensions,
            fixed_thumbnail_shape=fixed_thumbnail_shape,
            fixed_thumbnail_offset=fixed_thumbnail_offset,
            patch_size_wsi=patch_size_wsi,
            overlap=overlap,
            tissue_threshold=tissue_threshold,
        )
    return params, positions, patch_size_wsi, stride, fixed_mpp, used_fallback


def extract_registered_pair(
    registration_json,
    output_dir,
    target_mpp=0.5,
    patch_size=2048,
    overlap=0.2,
    tissue_threshold=0.3,
    fallback_mpp=0.5,
    trust_slide_mpp=False,
    max_patches=None,
    overwrite=False,
):
    """Extract one aligned H&E/Unstain patch set and return metadata rows."""
    registration_json = Path(registration_json)
    output_dir = Path(output_dir)
    hne_dir = output_dir / "hne"
    unstain_dir = output_dir / "unstain"
    hne_dir.mkdir(parents=True, exist_ok=True)
    unstain_dir.mkdir(parents=True, exist_ok=True)

    params, positions, patch_size_wsi, stride, fixed_mpp, used_fallback = (
        prepare_patch_positions(
            registration_json,
            target_mpp=target_mpp,
            patch_size=patch_size,
            overlap=overlap,
            tissue_threshold=tissue_threshold,
            fallback_mpp=fallback_mpp,
            trust_slide_mpp=trust_slide_mpp,
        )
    )
    if max_patches is not None:
        positions = positions[:max_patches]

    moving_to_fixed = np.asarray(
        params["transforms"]["moving_hne_to_fixed_unstain_fullres_3x3"],
        dtype=np.float64,
    )
    fixed_to_moving = np.linalg.inv(moving_to_fixed)
    stem = registration_json.name.removesuffix("_registration.json")
    rows = []

    with openslide.OpenSlide(params["files"]["moving_hne"]) as moving_slide, \
         openslide.OpenSlide(params["files"]["fixed_unstain"]) as fixed_slide:
        for index, (x, y, tissue_ratio) in enumerate(positions):
            patch_name = f"{stem}_{index:05d}_x{x}_y{y}.png"
            hne_path = hne_dir / patch_name
            unstain_path = unstain_dir / patch_name
            center = np.array([[x + patch_size_wsi / 2, y + patch_size_wsi / 2]])
            moving_center = transform_points(center, fixed_to_moving)[0]

            if not overwrite and hne_path.is_file() and unstain_path.is_file():
                status = "skipped"
            else:
                unstain_patch = _extract_fixed_patch(
                    fixed_slide, x, y, patch_size_wsi, patch_size
                )
                hne_patch, _ = _extract_aligned_moving_patch(
                    moving_slide,
                    x,
                    y,
                    patch_size_wsi,
                    patch_size,
                    fixed_to_moving,
                )
                if hne_patch is None:
                    continue
                Image.fromarray(unstain_patch).save(unstain_path, compress_level=3)
                Image.fromarray(hne_patch).save(hne_path, compress_level=3)
                status = "completed"

            rows.append(
                {
                    "slide": stem,
                    "patch_name": patch_name,
                    "fixed_unstain_x": x,
                    "fixed_unstain_y": y,
                    "moving_hne_center_x": float(moving_center[0]),
                    "moving_hne_center_y": float(moving_center[1]),
                    "patch_size_wsi": patch_size_wsi,
                    "output_patch_size": patch_size,
                    "target_mpp": target_mpp,
                    "fixed_mpp_used": fixed_mpp,
                    "used_fallback_mpp": used_fallback,
                    "common_tissue_ratio": tissue_ratio,
                    "status": status,
                }
            )

    metadata_path = output_dir / "metadata" / f"{stem}_patches.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with metadata_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows, {
        "slide": stem,
        "valid_positions": len(positions),
        "saved_or_skipped": len(rows),
        "patch_size_wsi": patch_size_wsi,
        "stride_wsi": stride,
        "fixed_mpp_used": fixed_mpp,
        "used_fallback_mpp": used_fallback,
    }


def extract_patch_dataset(
    registration_jsons,
    output_dir,
    target_mpp=0.5,
    patch_size=2048,
    overlap=0.2,
    tissue_threshold=0.3,
    fallback_mpp=0.5,
    trust_slide_mpp=False,
    max_patches_per_slide=None,
    overwrite=False,
    continue_on_error=True,
):
    """Batch extraction with one paired filename in the hne/ and unstain/ folders."""
    output_dir = Path(output_dir)
    summaries = []
    for json_path in tqdm(sorted(map(Path, registration_jsons)), desc="Patch dataset"):
        try:
            _, summary = extract_registered_pair(
                json_path,
                output_dir,
                target_mpp=target_mpp,
                patch_size=patch_size,
                overlap=overlap,
                tissue_threshold=tissue_threshold,
                fallback_mpp=fallback_mpp,
                trust_slide_mpp=trust_slide_mpp,
                max_patches=max_patches_per_slide,
                overwrite=overwrite,
            )
            summary["status"] = "completed"
            summaries.append(summary)
        except Exception as error:
            summaries.append(
                {"slide": json_path.stem, "status": "error", "error": str(error)}
            )
            if not continue_on_error:
                raise

    summary_path = output_dir / "extraction_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    if summaries:
        fieldnames = sorted({key for row in summaries for key in row})
        with summary_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
    return summaries
