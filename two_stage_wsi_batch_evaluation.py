from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from two_stage_inference import infer_wsi
from wsi_publication_evaluation import (
    METRIC_DIRECTIONS,
    PRIMARY_METRICS,
    DeepFeatureEvaluator,
    evaluate_wsi_tiles,
    read_slide_at_mpp,
    register_real_hne_to_unstain,
    summarize_cohort_metrics,
    summarize_distribution_features,
)


def _thumbnail(rgb, maximum=768):
    height, width = rgb.shape[:2]
    scale = min(1.0, float(maximum) / max(height, width))
    if scale == 1.0:
        return rgb
    import cv2

    return cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _seam_discontinuity(rgb, stride, tissue_gray_threshold=245):
    """Measure tile-boundary RGB jumps relative to nearby tissue transitions."""
    height, width = rgb.shape[:2]
    weighted = {"seam_sum": 0.0, "seam_n": 0, "control_sum": 0.0, "control_n": 0}

    def accumulate(axis, index, prefix):
        if axis == 1:
            left, right = rgb[:, index - 1], rgb[:, index]
        else:
            left, right = rgb[index - 1], rgb[index]
        mask = (
            (left.astype(np.float32).mean(axis=1) < tissue_gray_threshold)
            & (right.astype(np.float32).mean(axis=1) < tissue_gray_threshold)
        )
        count = int(mask.sum())
        if count:
            jump = float(
                np.abs(left[mask].astype(np.float32) - right[mask].astype(np.float32)).mean()
            )
            weighted[f"{prefix}_sum"] += jump * count
            weighted[f"{prefix}_n"] += count

    for axis, length in ((1, width), (0, height)):
        for boundary in range(int(stride), length, int(stride)):
            accumulate(axis, boundary, "seam")
            for offset in (-64, -32, -16, 16, 32, 64):
                control = boundary + offset
                if 1 <= control < length:
                    accumulate(axis, control, "control")

    seam = weighted["seam_sum"] / max(weighted["seam_n"], 1)
    control = weighted["control_sum"] / max(weighted["control_n"], 1)
    return {
        "seam_jump": seam,
        "nearby_jump": control,
        "seam_ratio": seam / max(control, 1e-12),
    }


def _save_slide_comparison(
    slide,
    unstain,
    generated,
    registered_hne,
    tissue_mask,
    tile_metrics,
    output_path,
):
    generated_rows = tile_metrics[tile_metrics["comparison"].eq("Virtual H&E")]
    best = generated_rows.sort_values("tissue_fraction", ascending=False).iloc[0]
    x, y = int(best["tile_x"]), int(best["tile_y"])
    size = 512
    region = np.s_[y:y + size, x:x + size]
    error = np.abs(
        generated[region].astype(np.float32) - registered_hne[region].astype(np.float32)
    ).mean(axis=2) / 255.0

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    top = (
        (unstain, "Input Unstain"),
        (generated, "Virtual H&E"),
        (registered_hne, "Registered real H&E"),
    )
    for axis, (image, title) in zip(axes[0, :3], top):
        axis.imshow(image)
        axis.set_title(title)
    whole_error = np.abs(
        generated.astype(np.float32) - registered_hne.astype(np.float32)
    ).mean(axis=2) / 255.0
    artist = axes[0, 3].imshow(
        np.ma.masked_where(~tissue_mask, whole_error), cmap="magma", vmin=0, vmax=0.5
    )
    axes[0, 3].set_title("Tissue RGB absolute error")
    fig.colorbar(artist, ax=axes[0, 3], fraction=0.046, pad=0.04)

    zoom = (
        (unstain[region], "Input zoom"),
        (generated[region], "Virtual H&E zoom"),
        (registered_hne[region], "Real H&E zoom"),
    )
    for axis, (image, title) in zip(axes[1, :3], zoom):
        axis.imshow(image)
        axis.set_title(title)
    artist = axes[1, 3].imshow(
        np.ma.masked_where(~tissue_mask[region], error),
        cmap="magma", vmin=0, vmax=0.5,
    )
    axes[1, 3].set_title(f"Matched error, x={x}, y={y}")
    fig.colorbar(artist, ax=axes[1, 3], fraction=0.046, pad=0.04)
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle(f"{slide} — 2.0 MPP", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_cohort_overview(rows, output_path):
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4 * len(rows)), squeeze=False)
    for row_index, row in enumerate(rows):
        for column, (key, title) in enumerate((
            ("unstain", "Input Unstain"),
            ("generated", "Virtual H&E"),
            ("reference", "Registered real H&E"),
        )):
            axes[row_index, column].imshow(row[key])
            axes[row_index, column].set_title(
                f"{row['slide']}\n{title}" if column == 0 else title
            )
            axes[row_index, column].axis("off")
    fig.suptitle("All held-out WSI comparisons at 2.0 MPP", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_cohort_metric_figure(summary, output_path):
    metrics = [metric for metric in PRIMARY_METRICS if metric in set(summary["metric"])]
    columns = 3
    rows = int(np.ceil(len(metrics) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 4.2 * rows), squeeze=False)
    order = ["Unstain baseline", "Virtual H&E"]
    colors = ["#9E9E9E", "#7B1FA2"]
    for axis, metric in zip(axes.flat, metrics):
        values = summary[summary["metric"].eq(metric)].set_index("comparison").loc[order]
        means = values["case_mean"].to_numpy()
        errors = np.vstack((
            means - values["ci95_low"].to_numpy(),
            values["ci95_high"].to_numpy() - means,
        ))
        axis.bar(order, means, yerr=errors, color=colors, capsize=5)
        direction = "↑" if METRIC_DIRECTIONS[metric] == "higher" else "↓"
        axis.set_title(f"{metric} {direction}")
        axis.tick_params(axis="x", rotation=12)
        axis.grid(axis="y", alpha=0.25)
    for axis in axes.flat[len(metrics):]:
        axis.axis("off")
    fig.suptitle("Held-out case-level metrics: mean and bootstrap 95% CI", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_heldout_wsi_evaluation(
    manifest,
    structure,
    colorizer,
    metadata,
    output_dir,
    *,
    reuse_existing=True,
    inference_batch_size=4,
    inference_tile_size=512,
    inference_overlap=256,
    inference_jpeg_quality=95,
    deep_batch_size=8,
    tile_size=512,
    stride=512,
    minimum_tissue_fraction=0.10,
    minimum_registration_iou=0.90,
    bootstrap_iterations=5000,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_slide_dir = output_dir / "per_slide"
    per_slide_dir.mkdir(exist_ok=True)
    deep_evaluator = DeepFeatureEvaluator(metadata["device"], deep_batch_size)
    all_metrics = []
    run_rows = []
    cohort_overview = []
    distribution_payloads = defaultdict(
        lambda: {"candidate_features": [], "reference_features": []}
    )

    ready_manifest = manifest[manifest["ready"]].reset_index(drop=True)
    for row_index, row in ready_manifest.iterrows():
        slide = row["slide"]
        print(f"[{row_index + 1}/{len(ready_manifest)}] {slide}")
        slide_dir = per_slide_dir / slide
        slide_dir.mkdir(exist_ok=True)
        generated_path = slide_dir / f"{slide}_virtual_HnE_2mpp.tiff"
        registered_path = slide_dir / f"{slide}_real_HnE_registered_2mpp.tiff"
        comparison_path = slide_dir / f"{slide}_comparison.png"
        try:
            if not (reuse_existing and generated_path.is_file()):
                infer_wsi(
                    row["unstain_path"], generated_path, structure, colorizer, metadata,
                    tile_size=inference_tile_size, overlap=inference_overlap,
                    batch_size=inference_batch_size,
                    fallback_mpp=0.5, minimum_tissue_fraction=0.002,
                    jpeg_quality=inference_jpeg_quality,
                )
            unstain, _, registered, valid_mask, registration = register_real_hne_to_unstain(
                row["unstain_path"], row["hne_path"], row["registration_path"],
                metadata["target_mpp"], output_path=registered_path, fallback_mpp=0.5,
            )
            generated, _ = read_slide_at_mpp(
                generated_path, metadata["target_mpp"], metadata["target_mpp"]
            )
            seam_metrics = _seam_discontinuity(
                generated, inference_tile_size - inference_overlap
            )
            tile_metrics, tissue_mask, feature_payload = evaluate_wsi_tiles(
                unstain, generated, registered, valid_mask,
                slide_name=slide, tile_size=tile_size, stride=stride,
                minimum_tissue_fraction=minimum_tissue_fraction,
                deep_evaluator=deep_evaluator,
            )
            tile_metrics.to_csv(slide_dir / f"{slide}_tile_metrics.csv", index=False)
            all_metrics.append(tile_metrics)
            for comparison, payload in feature_payload.items():
                distribution_payloads[comparison]["candidate_features"].append(
                    payload["candidate_features"]
                )
                distribution_payloads[comparison]["reference_features"].append(
                    payload["reference_features"]
                )
            _save_slide_comparison(
                slide, unstain, generated, registered, tissue_mask,
                tile_metrics, comparison_path,
            )
            cohort_overview.append({
                "slide": slide,
                "unstain": _thumbnail(unstain),
                "generated": _thumbnail(generated),
                "reference": _thumbnail(registered),
            })
            quality = registration.get("registration_quality", {})
            mask_iou = quality.get("mask_iou_after", np.nan)
            run_rows.append({
                "case_id": row["case_id"], "slide": slide, "status": "ok",
                "n_tiles": int(tile_metrics["comparison"].eq("Virtual H&E").sum()),
                "mask_iou_after": mask_iou,
                "ecc_score": quality.get("ecc_score", np.nan),
                "registration_qc_pass": bool(
                    np.isfinite(mask_iou) and mask_iou >= minimum_registration_iou
                ),
                "valid_fraction": registration.get("valid_fraction", np.nan),
                "inference_tile_size": inference_tile_size,
                "inference_overlap": inference_overlap,
                "inference_stride": inference_tile_size - inference_overlap,
                **seam_metrics,
                "generated_path": str(generated_path),
                "registered_hne_path": str(registered_path),
                "comparison_path": str(comparison_path),
                "error": "",
            })
        except Exception as error:
            run_rows.append({
                "case_id": row["case_id"], "slide": slide, "status": "failed",
                "n_tiles": 0, "mask_iou_after": np.nan, "ecc_score": np.nan,
                "registration_qc_pass": False,
                "valid_fraction": np.nan, "generated_path": str(generated_path),
                "inference_tile_size": inference_tile_size,
                "inference_overlap": inference_overlap,
                "inference_stride": inference_tile_size - inference_overlap,
                "seam_jump": np.nan, "nearby_jump": np.nan, "seam_ratio": np.nan,
                "registered_hne_path": str(registered_path),
                "comparison_path": str(comparison_path),
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"  FAILED: {type(error).__name__}: {error}")
        pd.DataFrame(run_rows).to_csv(output_dir / "run_manifest.csv", index=False)
        gc.collect()

    if not all_metrics:
        raise RuntimeError("No held-out WSI was evaluated successfully")
    tile_metrics = pd.concat(all_metrics, ignore_index=True)
    slide_metrics, case_metrics, summary, improvement = summarize_cohort_metrics(
        tile_metrics, bootstrap_iterations=bootstrap_iterations, seed=42
    )
    successful_runs = pd.DataFrame(run_rows).query("status == 'ok'")
    qc_slides = successful_runs.loc[
        successful_runs["registration_qc_pass"], "slide"
    ].tolist()
    qc_tile_metrics = tile_metrics[tile_metrics["slide"].isin(qc_slides)].copy()
    qc_slide_metrics, qc_case_metrics, qc_summary, qc_improvement = summarize_cohort_metrics(
        qc_tile_metrics, bootstrap_iterations=bootstrap_iterations, seed=42
    )
    distribution = summarize_distribution_features(distribution_payloads, seed=42)

    outputs = {
        "run_manifest": output_dir / "run_manifest.csv",
        "tile_metrics": output_dir / "all_tile_metrics.csv",
        "slide_metrics": output_dir / "slide_metrics.csv",
        "case_metrics": output_dir / "case_metrics.csv",
        "cohort_summary": output_dir / "cohort_metric_summary.csv",
        "cohort_improvement": output_dir / "cohort_metric_improvement.csv",
        "qc_slide_metrics": output_dir / "qc_passed_slide_metrics.csv",
        "qc_case_metrics": output_dir / "qc_passed_case_metrics.csv",
        "qc_cohort_summary": output_dir / "qc_passed_cohort_metric_summary.csv",
        "qc_cohort_improvement": output_dir / "qc_passed_cohort_metric_improvement.csv",
        "distribution_metrics": output_dir / "distribution_metrics.csv",
        "cohort_overview": output_dir / "cohort_wsi_overview.png",
        "cohort_metric_figure": output_dir / "cohort_metric_summary.png",
    }
    tile_metrics.to_csv(outputs["tile_metrics"], index=False)
    slide_metrics.to_csv(outputs["slide_metrics"], index=False)
    case_metrics.to_csv(outputs["case_metrics"], index=False)
    summary.to_csv(outputs["cohort_summary"], index=False)
    improvement.to_csv(outputs["cohort_improvement"], index=False)
    qc_slide_metrics.to_csv(outputs["qc_slide_metrics"], index=False)
    qc_case_metrics.to_csv(outputs["qc_case_metrics"], index=False)
    qc_summary.to_csv(outputs["qc_cohort_summary"], index=False)
    qc_improvement.to_csv(outputs["qc_cohort_improvement"], index=False)
    distribution.to_csv(outputs["distribution_metrics"], index=False)
    _save_cohort_overview(cohort_overview, outputs["cohort_overview"])
    _save_cohort_metric_figure(summary, outputs["cohort_metric_figure"])
    return {
        "run_manifest": pd.DataFrame(run_rows),
        "tile_metrics": tile_metrics,
        "slide_metrics": slide_metrics,
        "case_metrics": case_metrics,
        "cohort_summary": summary,
        "cohort_improvement": improvement,
        "qc_cohort_summary": qc_summary,
        "qc_cohort_improvement": qc_improvement,
        "distribution_metrics": distribution,
        "outputs": outputs,
    }
