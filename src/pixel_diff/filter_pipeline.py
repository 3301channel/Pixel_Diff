"""Optional three-level false-positive filtering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pixel_diff.models import DifferenceRegion, PixelDiffConfig
from pixel_diff.region_utils import renumber_regions
from pixel_diff.regions import filter_locally_similar_regions
from pixel_diff.similarity import best_ssim_for_region, best_ssim_for_region_cached


@dataclass(frozen=True)
class FilterPipelineResult:
    regions: list[DifferenceRegion]
    metrics: dict[str, int | float | str]
    annotations: list[dict[str, Any]]


def apply_multilevel_filters(
    regions: list[DifferenceRegion],
    aligned_bgr: np.ndarray,
    template_bgr: np.ndarray,
    scan_binary: np.ndarray,
    template_binary: np.ndarray,
    scan_path: str | Path,
    template_path: str | Path,
    page: int,
    config: PixelDiffConfig,
    scan_gray: np.ndarray | None = None,
    template_gray: np.ndarray | None = None,
) -> FilterPipelineResult:
    scan_foreground = (scan_binary == 0).astype(np.uint8)
    template_foreground = (template_binary == 0).astype(np.uint8)
    prefiltered_regions, prefilter_removed = _filter_and_merge_level1_residuals(
        regions,
        aligned_bgr,
        template_bgr,
        scan_foreground,
        template_foreground,
        config,
    )
    level1_regions = filter_locally_similar_regions(
        prefiltered_regions,
        scan_binary=scan_binary,
        template_binary=template_binary,
        config=config,
    )
    level1_removed = len(regions) - len(level1_regions)
    if not config.multilevel_filter_enabled:
        return FilterPipelineResult(
            regions=renumber_regions(level1_regions),
            metrics=_metrics(
                input_regions=len(regions),
                level1_removed=level1_removed,
            ),
            annotations=[],
        )

    level2_regions, ssim_metrics = _filter_by_ssim(
        level1_regions,
        aligned_bgr,
        template_bgr,
        config,
        scan_gray=scan_gray,
        template_gray=template_gray,
    )
    annotations = _collect_text_annotations(
        level2_regions,
        scan_path=scan_path,
        template_path=template_path,
        page=page,
        config=config,
    )
    return FilterPipelineResult(
        regions=renumber_regions(level2_regions),
        metrics={
            **_metrics(
                input_regions=len(regions),
                level1_removed=level1_removed,
                level2_removed=len(level1_regions) - len(level2_regions),
                level3_annotations=len(annotations),
            ),
            **ssim_metrics,
        },
        annotations=annotations,
    )


def _filter_and_merge_level1_residuals(
    regions: list[DifferenceRegion],
    aligned_bgr: np.ndarray,
    template_bgr: np.ndarray,
    scan_foreground: np.ndarray,
    template_foreground: np.ndarray,
    config: PixelDiffConfig,
) -> tuple[list[DifferenceRegion], int]:
    if not regions or not config.multilevel_filter_enabled:
        return regions, 0

    merged_regions, merge_removed = _merge_same_line_small_residuals(regions, config)
    color_filtered_regions, color_removed = _remove_colored_residuals(
        merged_regions,
        aligned_bgr,
        template_bgr,
        config,
    )
    filtered_regions, isolated_removed = _remove_isolated_small_residuals(
        color_filtered_regions,
        scan_foreground,
        template_foreground,
        config,
    )
    return filtered_regions, merge_removed + color_removed + isolated_removed


def _remove_colored_residuals(
    regions: list[DifferenceRegion],
    aligned_bgr: np.ndarray,
    template_bgr: np.ndarray,
    config: PixelDiffConfig,
) -> tuple[list[DifferenceRegion], int]:
    if not config.colored_residual_filter_enabled:
        return regions, 0

    kept = []
    removed = 0
    for region in regions:
        color_ratio = max(
            _colored_pixel_ratio(region, aligned_bgr, config.colored_residual_padding),
            _colored_pixel_ratio(region, template_bgr, config.colored_residual_padding),
        )
        if color_ratio >= config.colored_residual_min_ratio:
            removed += 1
            continue
        kept.append(region)
    return kept, removed


def _merge_same_line_small_residuals(
    regions: list[DifferenceRegion],
    config: PixelDiffConfig,
) -> tuple[list[DifferenceRegion], int]:
    if not config.same_line_merge_filter_enabled:
        return regions, 0

    merged: list[DifferenceRegion] = []
    consumed: set[int] = set()
    ordered = sorted(enumerate(regions), key=lambda item: (item[1].y, item[1].x))
    for index, region in ordered:
        if index in consumed:
            continue
        current = region
        changed = True
        while changed:
            changed = False
            for other_index, other in ordered:
                if other_index == index or other_index in consumed:
                    continue
                if _should_merge_same_line(current, other, config):
                    current = _union_region(current, other)
                    consumed.add(other_index)
                    changed = True
        merged.append(current)
        consumed.add(index)

    return _sort_regions(merged), len(regions) - len(merged)


def _should_merge_same_line(
    first: DifferenceRegion,
    second: DifferenceRegion,
    config: PixelDiffConfig,
) -> bool:
    if min(first.area, second.area) > config.same_line_merge_small_area:
        return False
    if abs(_center_y(first) - _center_y(second)) > config.same_line_merge_max_center_y_delta:
        return False
    return _horizontal_gap(first, second) <= config.same_line_merge_max_gap


def _remove_isolated_small_residuals(
    regions: list[DifferenceRegion],
    scan_foreground: np.ndarray,
    template_foreground: np.ndarray,
    config: PixelDiffConfig,
) -> tuple[list[DifferenceRegion], int]:
    if not config.isolated_residual_filter_enabled or len(regions) <= 1:
        return regions, 0

    kept = []
    removed = 0
    for region in regions:
        if not _is_small_isolated_candidate(region, config):
            kept.append(region)
            continue
        if _foreground_density(region, scan_foreground, template_foreground) > (
            config.isolated_residual_max_density
        ):
            kept.append(region)
            continue
        nearest_distance = min(
            _center_distance(region, other)
            for other in regions
            if other is not region
        )
        if nearest_distance > config.isolated_residual_min_neighbor_distance:
            removed += 1
            continue
        kept.append(region)
    return kept, removed


def _is_small_isolated_candidate(region: DifferenceRegion, config: PixelDiffConfig) -> bool:
    return (
        region.area <= config.isolated_residual_max_area
        and region.width <= config.isolated_residual_max_width
        and region.height <= config.isolated_residual_max_height
    )


def _foreground_density(
    region: DifferenceRegion,
    scan_foreground: np.ndarray,
    template_foreground: np.ndarray,
) -> float:
    height, width = scan_foreground.shape[:2]
    x0 = max(0, region.x)
    y0 = max(0, region.y)
    x1 = min(width, region.x + region.width)
    y1 = min(height, region.y + region.height)
    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        return 0.0
    foreground = int(scan_foreground[y0:y1, x0:x1].sum() + template_foreground[y0:y1, x0:x1].sum())
    return float(foreground / (area * 2))


def _colored_pixel_ratio(region: DifferenceRegion, image_bgr: np.ndarray, padding: int) -> float:
    height, width = image_bgr.shape[:2]
    x0 = max(0, region.x - padding)
    y0 = max(0, region.y - padding)
    x1 = min(width, region.x + region.width + padding)
    y1 = min(height, region.y + region.height + padding)
    roi = image_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturated = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
    red = saturated & ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
    blue = saturated & ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 130))
    return float((red | blue).mean())


def _filter_by_ssim(
    regions: list[DifferenceRegion],
    aligned_bgr: np.ndarray,
    template_bgr: np.ndarray,
    config: PixelDiffConfig,
    *,
    scan_gray: np.ndarray | None = None,
    template_gray: np.ndarray | None = None,
) -> tuple[list[DifferenceRegion], dict[str, int | float]]:
    if not config.ssim_filter_enabled or not regions:
        return regions, {"filter_ssim_checked": 0, "filter_ssim_max_score": 0.0}

    if scan_gray is None:
        scan_gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    if template_gray is None:
        template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    kept: list[DifferenceRegion] = []
    checked = 0
    max_score = 0.0
    for region in regions:
        if region.area < config.ssim_filter_min_region_area:
            kept.append(region)
            continue
        checked += 1
        ssim_search = (
            best_ssim_for_region_cached
            if config.ssim_cached_template_stats_enabled
            else best_ssim_for_region
        )
        score = ssim_search(
            region,
            scan_gray,
            template_gray,
            padding=config.ssim_filter_padding,
            search_radius=config.ssim_filter_search_radius,
            stop_at=(
                config.ssim_filter_threshold
                if config.ssim_early_exit_enabled
                else None
            ),
        )
        max_score = max(max_score, score)
        if score >= config.ssim_filter_threshold:
            continue
        kept.append(region)
    return kept, {"filter_ssim_checked": checked, "filter_ssim_max_score": max_score}


def _collect_text_annotations(
    regions: list[DifferenceRegion],
    scan_path: str | Path,
    template_path: str | Path,
    page: int,
    config: PixelDiffConfig,
) -> list[dict[str, Any]]:
    if not config.text_annotation_enabled or not regions:
        return []
    try:
        from pixel_diff.text_layer import compare_text_regions
    except ImportError:
        return []
    return compare_text_regions(regions, scan_path, template_path, page, config.dpi)


def _metrics(
    input_regions: int,
    level1_removed: int,
    level2_removed: int = 0,
    level3_annotations: int = 0,
) -> dict[str, int | float | str]:
    return {
        "filter_input_regions": input_regions,
        "filter_level1_removed": level1_removed,
        "filter_level2_removed": level2_removed,
        "filter_level3_annotations": level3_annotations,
        "filter_ssim_checked": 0,
        "filter_ssim_max_score": 0.0,
    }


def _sort_regions(regions: list[DifferenceRegion]) -> list[DifferenceRegion]:
    return sorted(regions, key=lambda region: (region.y, region.x, -region.area))


def _union_region(first: DifferenceRegion, second: DifferenceRegion) -> DifferenceRegion:
    x0 = min(first.x, second.x)
    y0 = min(first.y, second.y)
    x1 = max(first.x + first.width, second.x + second.width)
    y1 = max(first.y + first.height, second.y + second.height)
    return DifferenceRegion(
        id=first.id,
        x=x0,
        y=y0,
        width=x1 - x0,
        height=y1 - y0,
        area=max(first.area, second.area, float((x1 - x0) * (y1 - y0))),
    )


def _center_x(region: DifferenceRegion) -> float:
    return region.x + region.width / 2


def _center_y(region: DifferenceRegion) -> float:
    return region.y + region.height / 2


def _center_distance(first: DifferenceRegion, second: DifferenceRegion) -> float:
    return float(
        np.hypot(
            _center_x(first) - _center_x(second),
            _center_y(first) - _center_y(second),
        )
    )


def _horizontal_gap(first: DifferenceRegion, second: DifferenceRegion) -> int:
    first_right = first.x + first.width
    second_right = second.x + second.width
    if first_right < second.x:
        return second.x - first_right
    if second_right < first.x:
        return first.x - second_right
    return 0
