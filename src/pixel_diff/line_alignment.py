"""Text-line centroid alignment for residual vertical row drift."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pixel_diff.models import PixelDiffConfig


@dataclass(frozen=True)
class LineCentroidAlignmentResult:
    aligned_bgr: np.ndarray
    applied: bool
    scan_lines: int
    template_lines: int
    matched_pairs: int
    max_abs_offset: float
    horizontal_applied: bool = False
    horizontal_anchors: int = 0
    max_abs_horizontal_offset: float = 0.0
    matched_line_centers: tuple[int, ...] = ()


def align_text_lines_by_centroid_bgr(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    config: PixelDiffConfig,
) -> LineCentroidAlignmentResult:
    """Apply optional Y-only text-line centroid compensation after global alignment."""

    if not config.line_centroid_alignment:
        return LineCentroidAlignmentResult(
            aligned_bgr=scan_bgr,
            applied=False,
            scan_lines=0,
            template_lines=0,
            matched_pairs=0,
            max_abs_offset=0.0,
        )

    height, width = template_bgr.shape[:2]
    centers_scan = _extract_line_centroids(scan_bgr, config)
    centers_template = _extract_line_centroids(template_bgr, config)
    matched_pairs = _match_line_centroids(
        centers_scan=centers_scan,
        centers_template=centers_template,
        max_drift=config.line_centroid_max_drift,
        config=config,
    )
    if not matched_pairs:
        return LineCentroidAlignmentResult(
            aligned_bgr=scan_bgr,
            applied=False,
            scan_lines=len(centers_scan),
            template_lines=len(centers_template),
            matched_pairs=0,
            max_abs_offset=0.0,
        )

    max_abs_offset = max(abs(y_template - y_scan) for y_template, y_scan in matched_pairs)
    aligned = scan_bgr
    vertical_applied = max_abs_offset > 0
    if vertical_applied:
        anchor_y = [0.0]
        anchor_offsets = [0.0]
        for y_template, y_scan in matched_pairs:
            anchor_y.append(float(y_template))
            anchor_offsets.append(float(y_template - y_scan))
        anchor_y.append(float(height - 1))
        anchor_offsets.append(0.0)

        all_y = np.arange(height, dtype=np.float32)
        dy = np.interp(all_y, anchor_y, anchor_offsets).astype(np.float32)
        map_y = all_y - dy
        map_x = np.arange(width, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(map_x, map_y)
        aligned = cv2.remap(
            scan_bgr,
            grid_x,
            grid_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    aligned, horizontal_offsets = _apply_horizontal_line_alignment(
        aligned,
        template_bgr,
        [y_template for y_template, _y_scan in matched_pairs],
        config,
    )
    return LineCentroidAlignmentResult(
        aligned_bgr=aligned,
        applied=vertical_applied or bool(horizontal_offsets),
        scan_lines=len(centers_scan),
        template_lines=len(centers_template),
        matched_pairs=len(matched_pairs),
        max_abs_offset=float(max_abs_offset),
        horizontal_applied=bool(horizontal_offsets),
        horizontal_anchors=len(horizontal_offsets),
        max_abs_horizontal_offset=float(
            max((abs(offset) for _center, offset in horizontal_offsets), default=0)
        ),
        matched_line_centers=tuple(y_template for y_template, _y_scan in matched_pairs),
    )


def _apply_horizontal_line_alignment(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    line_centers: list[int],
    config: PixelDiffConfig,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if not config.line_horizontal_alignment or config.line_horizontal_max_shift == 0:
        return scan_bgr, []

    scan_foreground = _binary_foreground(scan_bgr)
    template_foreground = _binary_foreground(template_bgr)
    offsets: list[tuple[int, int]] = []
    for center_y in line_centers:
        estimate = _estimate_horizontal_line_offset(
            scan_foreground,
            template_foreground,
            center_y=center_y,
            band_half_height=config.line_horizontal_band_half_height,
            max_shift=config.line_horizontal_max_shift,
            min_iou=config.line_horizontal_min_iou,
            min_improvement=config.line_horizontal_min_improvement,
        )
        if estimate is not None:
            correction, _best_iou, _improvement = estimate
            offsets.append((center_y, correction))

    if not offsets:
        return scan_bgr, []

    height, width = scan_bgr.shape[:2]
    anchor_y = np.array([0, *[center for center, _offset in offsets], height - 1], dtype=np.float32)
    anchor_dx = np.array([0, *[offset for _center, offset in offsets], 0], dtype=np.float32)
    all_y = np.arange(height, dtype=np.float32)
    dx = np.interp(all_y, anchor_y, anchor_dx).astype(np.float32)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        all_y,
    )
    aligned = cv2.remap(
        scan_bgr,
        grid_x - dx[:, None],
        grid_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, offsets


def _estimate_horizontal_line_offset(
    scan_foreground: np.ndarray,
    template_foreground: np.ndarray,
    *,
    center_y: int,
    band_half_height: int,
    max_shift: int,
    min_iou: float,
    min_improvement: float,
) -> tuple[int, float, float] | None:
    height, width = scan_foreground.shape[:2]
    y0 = max(0, center_y - band_half_height)
    y1 = min(height, center_y + band_half_height + 1)
    scan_band = scan_foreground[y0:y1]
    template_band = template_foreground[y0:y1]
    if scan_band.size == 0 or template_band.size == 0:
        return None

    scores = {
        shift: _foreground_iou_for_horizontal_shift(scan_band, template_band, shift)
        for shift in range(-max_shift, max_shift + 1)
    }
    best_shift = max(scores, key=scores.__getitem__)
    best_iou = scores[best_shift]
    improvement = best_iou - scores[0]
    if best_shift == 0 or best_iou < min_iou or improvement < min_improvement:
        return None
    return -best_shift, best_iou, improvement


def _foreground_iou_for_horizontal_shift(
    scan_band: np.ndarray,
    template_band: np.ndarray,
    shift: int,
) -> float:
    if shift > 0:
        scan_view = scan_band[:, shift:]
        template_view = template_band[:, :-shift]
    elif shift < 0:
        scan_view = scan_band[:, :shift]
        template_view = template_band[:, -shift:]
    else:
        scan_view = scan_band
        template_view = template_band
    intersection = np.logical_and(scan_view, template_view).sum()
    union = np.logical_or(scan_view, template_view).sum()
    return 1.0 if union == 0 else float(intersection / union)


def _binary_foreground(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _extract_line_centroids(image_bgr: np.ndarray, config: PixelDiffConfig) -> list[int]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (config.line_centroid_row_dilate_width, config.line_centroid_row_dilate_height),
    )
    dilated = cv2.dilate(binary, kernel)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _, image_width = binary.shape[:2]
    min_width = image_width * config.line_centroid_min_width_ratio
    centroids = []
    for contour in contours:
        _x, y, width, height = cv2.boundingRect(contour)
        if (
            width > min_width
            and config.line_centroid_min_height < height < config.line_centroid_max_height
        ):
            centroids.append(y + height // 2)
    return sorted(centroids)


def _match_line_centroids(
    centers_scan: list[int],
    centers_template: list[int],
    max_drift: int,
    config: PixelDiffConfig | None = None,
) -> list[tuple[int, int]]:
    matched_pairs = []
    used_scan_centers: set[int] = set()
    for y_template in centers_template:
        nearest_scan = _nearest_center(y_template, centers_scan, max_drift)
        if nearest_scan is None or nearest_scan in used_scan_centers:
            continue
        nearest_template = _nearest_center(nearest_scan, centers_template, max_drift)
        if nearest_template != y_template:
            continue
        matched_pairs.append((y_template, nearest_scan))
        used_scan_centers.add(nearest_scan)
    if config is not None and config.line_centroid_consistency_filter:
        matched_pairs = _filter_monotonic_pairs(matched_pairs)
        matched_pairs = _filter_median_consistent_pairs(
            matched_pairs,
            tolerance=config.line_centroid_median_tolerance,
        )
    return matched_pairs


def _filter_monotonic_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(pairs) <= 1:
        return pairs

    filtered: list[tuple[int, int]] = []
    last_scan = -1
    for y_template, y_scan in sorted(pairs):
        if y_scan <= last_scan:
            continue
        filtered.append((y_template, y_scan))
        last_scan = y_scan
    return filtered


def _filter_median_consistent_pairs(
    pairs: list[tuple[int, int]],
    tolerance: int,
) -> list[tuple[int, int]]:
    if len(pairs) <= 1:
        return pairs

    offsets = np.array([y_template - y_scan for y_template, y_scan in pairs], dtype=np.float32)
    median_offset = float(np.median(offsets))
    return [
        pair
        for pair, offset in zip(pairs, offsets, strict=True)
        if abs(float(offset) - median_offset) <= tolerance
    ]


def _nearest_center(target: int, centers: list[int], max_drift: int) -> int | None:
    best_center = None
    best_distance = max_drift + 1
    for center in centers:
        distance = abs(target - center)
        if distance <= max_drift and distance < best_distance:
            best_center = center
            best_distance = distance
    return best_center
