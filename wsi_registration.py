"""Mask-based affine registration for nearly aligned whole-slide images."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openslide
from tqdm.auto import tqdm


@dataclass
class RegistrationResult:
    moving_image: np.ndarray
    fixed_image: np.ndarray
    aligned_image: np.ndarray
    moving_mask: np.ndarray
    fixed_mask: np.ndarray
    aligned_mask: np.ndarray
    warp_fixed_to_moving: np.ndarray
    matrix_moving_to_fixed_thumbnail: np.ndarray
    matrix_moving_to_fixed_fullres: np.ndarray
    moving_full_dimensions: tuple[int, int]
    fixed_full_dimensions: tuple[int, int]
    fixed_mask_threshold_offset: int
    ecc_score: float
    iou_before: float
    iou_after: float


def _load_thumbnail(slide_path, max_size):
    with openslide.OpenSlide(str(slide_path)) as slide:
        dimensions = slide.dimensions
        thumbnail = slide.get_thumbnail((max_size, max_size)).convert("RGB")
    return np.asarray(thumbnail), dimensions


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


def _pad_pair(moving, fixed, moving_value=255, fixed_value=255):
    target_height = max(moving.shape[0], fixed.shape[0])
    target_width = max(moving.shape[1], fixed.shape[1])
    moving_padded, moving_offset = _center_pad(
        moving, target_height, target_width, moving_value
    )
    fixed_padded, fixed_offset = _center_pad(
        fixed, target_height, target_width, fixed_value
    )
    return moving_padded, fixed_padded, moving_offset, fixed_offset


def create_tissue_mask(
    rgb_image, threshold_offset=0, min_component_fraction=0.0005
):
    """Create a stain-independent tissue mask from a white-background slide."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_threshold, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    sensitive_threshold = min(254, otsu_threshold + threshold_offset)
    mask = np.where(gray <= sensitive_threshold, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    cleaned = np.zeros_like(mask)
    minimum_area = max(16, int(mask.size * min_component_fraction))
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            cleaned[labels == label] = 255
    return cleaned


def mask_iou(mask1, mask2):
    foreground1 = mask1 > 0
    foreground2 = mask2 > 0
    union = np.logical_or(foreground1, foreground2).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(foreground1, foreground2).sum() / union)


def _mask_centroid(mask):
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        return mask.shape[1] / 2, mask.shape[0] / 2
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _homogeneous(affine):
    return np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float64)


def _fullres_matrix(
    moving_to_fixed_padded,
    moving_shape,
    fixed_shape,
    moving_dimensions,
    fixed_dimensions,
    moving_offset,
    fixed_offset,
):
    moving_height, moving_width = moving_shape[:2]
    fixed_height, fixed_width = fixed_shape[:2]
    moving_full_width, moving_full_height = moving_dimensions
    fixed_full_width, fixed_full_height = fixed_dimensions

    moving_full_to_thumb = np.array(
        [
            [moving_width / moving_full_width, 0.0, 0.0],
            [0.0, moving_height / moving_full_height, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    fixed_full_to_thumb = np.array(
        [
            [fixed_width / fixed_full_width, 0.0, 0.0],
            [0.0, fixed_height / fixed_full_height, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    moving_thumb_to_padded = np.array(
        [[1.0, 0.0, moving_offset[0]], [0.0, 1.0, moving_offset[1]], [0, 0, 1]]
    )
    fixed_thumb_to_padded = np.array(
        [[1.0, 0.0, fixed_offset[0]], [0.0, 1.0, fixed_offset[1]], [0, 0, 1]]
    )

    return (
        np.linalg.inv(fixed_full_to_thumb)
        @ np.linalg.inv(fixed_thumb_to_padded)
        @ moving_to_fixed_padded
        @ moving_thumb_to_padded
        @ moving_full_to_thumb
    )


def register_wsi_pair(
    moving_path,
    fixed_path,
    max_size=1000,
    fixed_threshold_offsets=(0, 2, 4, 6, 8, 10),
):
    """Align moving WSI to fixed WSI with scale, rotation, shear, and translation."""
    moving_raw, moving_dimensions = _load_thumbnail(moving_path, max_size)
    fixed_raw, fixed_dimensions = _load_thumbnail(fixed_path, max_size)
    moving_shape = moving_raw.shape
    fixed_shape = fixed_raw.shape

    moving, fixed, moving_offset, fixed_offset = _pad_pair(moving_raw, fixed_raw)
    moving_mask = create_tissue_mask(moving)
    moving_area = max(1, np.count_nonzero(moving_mask))
    fixed_mask_candidates = [
        (offset, create_tissue_mask(fixed, threshold_offset=offset))
        for offset in fixed_threshold_offsets
    ]
    fixed_threshold_offset, fixed_mask = min(
        fixed_mask_candidates,
        key=lambda candidate: abs(
            np.log(max(1, np.count_nonzero(candidate[1])) / moving_area)
        ),
    )

    moving_feature = cv2.GaussianBlur(
        moving_mask.astype(np.float32) / 255.0, (0, 0), 3
    )
    fixed_feature = cv2.GaussianBlur(
        fixed_mask.astype(np.float32) / 255.0, (0, 0), 3
    )

    moving_center = _mask_centroid(moving_mask)
    fixed_center = _mask_centroid(fixed_mask)
    warp = np.eye(2, 3, dtype=np.float32)
    # ECC's inverse-map warp maps fixed output coordinates to moving input coordinates.
    warp[0, 2] = moving_center[0] - fixed_center[0]
    warp[1, 2] = moving_center[1] - fixed_center[1]
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        500,
        1e-7,
    )

    try:
        ecc_score, warp = cv2.findTransformECC(
            fixed_feature,
            moving_feature,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error as error:
        name = Path(moving_path).name
        raise RuntimeError(f"ECC registration failed for {name}: {error}") from error

    height, width = fixed.shape[:2]
    inverse_flags = cv2.WARP_INVERSE_MAP
    aligned_image = cv2.warpAffine(
        moving,
        warp,
        (width, height),
        flags=cv2.INTER_LINEAR | inverse_flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    aligned_mask = cv2.warpAffine(
        moving_mask,
        warp,
        (width, height),
        flags=cv2.INTER_NEAREST | inverse_flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    fixed_to_moving = _homogeneous(warp)
    moving_to_fixed = np.linalg.inv(fixed_to_moving)
    fullres_matrix = _fullres_matrix(
        moving_to_fixed,
        moving_shape,
        fixed_shape,
        moving_dimensions,
        fixed_dimensions,
        moving_offset,
        fixed_offset,
    )

    return RegistrationResult(
        moving_image=moving,
        fixed_image=fixed,
        aligned_image=aligned_image,
        moving_mask=moving_mask,
        fixed_mask=fixed_mask,
        aligned_mask=aligned_mask,
        warp_fixed_to_moving=warp,
        matrix_moving_to_fixed_thumbnail=moving_to_fixed,
        matrix_moving_to_fixed_fullres=fullres_matrix,
        moving_full_dimensions=moving_dimensions,
        fixed_full_dimensions=fixed_dimensions,
        fixed_mask_threshold_offset=int(fixed_threshold_offset),
        ecc_score=float(ecc_score),
        iou_before=mask_iou(moving_mask, fixed_mask),
        iou_after=mask_iou(aligned_mask, fixed_mask),
    )


def transform_points(points, matrix):
    """Map Nx2 moving-image coordinates to fixed-image coordinates."""
    points = np.asarray(points, dtype=np.float64)
    homogeneous_points = np.column_stack([points, np.ones(len(points))])
    transformed = (np.asarray(matrix) @ homogeneous_points.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def affine_parameters(matrix):
    """Return an easy-to-read decomposition of a 2D affine matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    linear = matrix[:2, :2]
    return {
        "rotation_degrees": float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))),
        "scale_x": float(np.linalg.norm(linear[:, 0])),
        "scale_y": float(np.linalg.norm(linear[:, 1])),
        "translation_x": float(matrix[0, 2]),
        "translation_y": float(matrix[1, 2]),
    }


def save_registration_json(result, moving_path, fixed_path, output_path):
    """Save coordinate transforms and registration quality for one WSI pair."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "files": {
            "moving_hne": str(Path(moving_path).resolve()),
            "fixed_unstain": str(Path(fixed_path).resolve()),
        },
        "dimensions": {
            "moving_hne_full": {
                "width": result.moving_full_dimensions[0],
                "height": result.moving_full_dimensions[1],
            },
            "fixed_unstain_full": {
                "width": result.fixed_full_dimensions[0],
                "height": result.fixed_full_dimensions[1],
            },
            "registration_canvas": {
                "width": result.fixed_image.shape[1],
                "height": result.fixed_image.shape[0],
            },
        },
        "mask": {
            "fixed_unstain_threshold_offset": result.fixed_mask_threshold_offset,
        },
        "quality": {
            "mask_iou_before": result.iou_before,
            "mask_iou_after": result.iou_after,
            "ecc_score": result.ecc_score,
        },
        "transforms": {
            "fixed_to_moving_thumbnail_inverse_warp_2x3": (
                result.warp_fixed_to_moving.tolist()
            ),
            "moving_hne_to_fixed_unstain_thumbnail_3x3": (
                result.matrix_moving_to_fixed_thumbnail.tolist()
            ),
            "moving_hne_to_fixed_unstain_fullres_3x3": (
                result.matrix_moving_to_fixed_fullres.tolist()
            ),
        },
        "affine_parameters": {
            "thumbnail": affine_parameters(
                result.matrix_moving_to_fixed_thumbnail
            ),
            "fullres": affine_parameters(result.matrix_moving_to_fixed_fullres),
        },
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, ensure_ascii=False)
    return output_path


def plot_registration(result, moving_label="H&E", fixed_label="Unstain"):
    import matplotlib.pyplot as plt

    before = cv2.addWeighted(result.moving_image, 0.5, result.fixed_image, 0.5, 0)
    after = cv2.addWeighted(result.aligned_image, 0.5, result.fixed_image, 0.5, 0)
    mask_overlay = np.zeros((*result.fixed_mask.shape, 3), dtype=np.uint8)
    mask_overlay[result.aligned_mask > 0, 0] = 255
    mask_overlay[result.fixed_mask > 0, 2] = 255

    figure, axes = plt.subplots(2, 3, figsize=(18, 9))
    panels = [
        (result.moving_image, f"Moving: {moving_label}"),
        (result.fixed_image, f"Fixed: {fixed_label}"),
        (before, f"Before (IoU={result.iou_before:.4f})"),
        (result.aligned_image, f"Aligned {moving_label}"),
        (mask_overlay, f"Mask overlap (IoU={result.iou_after:.4f})"),
        (after, f"After (ECC={result.ecc_score:.4f})"),
    ]
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    return figure


def register_and_save_pairs(
    moving_paths,
    fixed_paths,
    output_dir,
    max_size=1000,
    overwrite=False,
    continue_on_error=True,
):
    """Register paired WSIs and save one JSON and diagnostic PNG per pair."""
    moving_paths = [Path(path) for path in moving_paths]
    fixed_paths = [Path(path) for path in fixed_paths]
    if len(moving_paths) != len(fixed_paths):
        raise ValueError("moving_paths and fixed_paths must have the same length")

    output_dir = Path(output_dir)
    parameter_dir = output_dir / "matrices"
    plot_dir = output_dir / "plots"
    parameter_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    pairs = list(zip(moving_paths, fixed_paths))
    for moving_path, fixed_path in tqdm(pairs, desc="WSI registration"):
        if moving_path.name != fixed_path.name:
            error = (
                f"Pair names do not match: {moving_path.name} != {fixed_path.name}"
            )
            summaries.append({"name": moving_path.stem, "status": "error", "error": error})
            if not continue_on_error:
                raise ValueError(error)
            continue

        stem = moving_path.stem
        json_path = parameter_dir / f"{stem}_registration.json"
        plot_path = plot_dir / f"{stem}_registration.png"
        if not overwrite and json_path.is_file() and plot_path.is_file():
            with json_path.open(encoding="utf-8") as input_file:
                saved = json.load(input_file)
            summaries.append(
                {
                    "name": stem,
                    "status": "skipped",
                    "mask_threshold_offset": saved["mask"][
                        "fixed_unstain_threshold_offset"
                    ],
                    "iou_before": saved["quality"]["mask_iou_before"],
                    "iou_after": saved["quality"]["mask_iou_after"],
                    "ecc_score": saved["quality"]["ecc_score"],
                    "matrix_json": str(json_path),
                    "plot_png": str(plot_path),
                }
            )
            continue

        try:
            result = register_wsi_pair(
                moving_path=moving_path,
                fixed_path=fixed_path,
                max_size=max_size,
            )
            save_registration_json(result, moving_path, fixed_path, json_path)
            figure = plot_registration(
                result, moving_label="H&E", fixed_label="Unstain"
            )
            figure.savefig(plot_path, dpi=140, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(figure)
            summaries.append(
                {
                    "name": stem,
                    "status": "completed",
                    "mask_threshold_offset": result.fixed_mask_threshold_offset,
                    "iou_before": result.iou_before,
                    "iou_after": result.iou_after,
                    "ecc_score": result.ecc_score,
                    "matrix_json": str(json_path),
                    "plot_png": str(plot_path),
                }
            )
        except Exception as error:
            summaries.append(
                {"name": stem, "status": "error", "error": str(error)}
            )
            if not continue_on_error:
                raise

    summary_path = output_dir / "registration_summary.csv"
    fieldnames = sorted({key for row in summaries for key in row})
    with summary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return summaries
