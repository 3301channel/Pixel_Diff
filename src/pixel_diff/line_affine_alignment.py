"""Confidence-gated horizontal affine correction for individual text lines."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pixel_diff.models import PixelDiffConfig


@dataclass(frozen=True)
class LineAffineAlignmentResult:
    aligned_bgr: np.ndarray
    applied_lines: int
    checked_lines: int
    before_iou: float
    after_iou: float
    max_scale_delta: float
    max_displacement: float


def align_text_lines_affine_bgr(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    line_centers: list[int],
    config: PixelDiffConfig,
) -> LineAffineAlignmentResult:
    if not config.line_affine_alignment_enabled or not line_centers:
        return LineAffineAlignmentResult(scan_bgr, 0, 0, 0.0, 0.0, 0.0, 0.0)

    scan_fg = _foreground(scan_bgr)
    template_fg = _foreground(template_bgr)
    output = scan_bgr.copy()
    applied = 0
    before_scores: list[float] = []
    after_scores: list[float] = []
    scale_deltas: list[float] = []
    displacements: list[float] = []
    height, width = template_fg.shape

    for center_y in line_centers:
        y0 = max(0, center_y - config.line_affine_band_half_height)
        y1 = min(height, center_y + config.line_affine_band_half_height + 1)
        scan_band = scan_fg[y0:y1]
        template_band = template_fg[y0:y1]
        estimate = _estimate_line_affine(scan_band, template_band, config)
        if estimate is None:
            continue
        slope, intercept, before_iou, after_iou = estimate
        x = np.arange(width, dtype=np.float32)
        correction = slope * x + intercept
        map_x = x - correction
        grid_x, grid_y = np.meshgrid(
            map_x,
            np.arange(y1 - y0, dtype=np.float32),
        )
        output[y0:y1] = cv2.remap(
            scan_bgr[y0:y1],
            grid_x,
            grid_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        applied += 1
        before_scores.append(before_iou)
        after_scores.append(after_iou)
        scale_deltas.append(abs(slope))
        displacements.append(float(np.max(np.abs(correction))))

    return LineAffineAlignmentResult(
        output,
        applied,
        len(line_centers),
        float(np.mean(before_scores)) if before_scores else 0.0,
        float(np.mean(after_scores)) if after_scores else 0.0,
        max(scale_deltas, default=0.0),
        max(displacements, default=0.0),
    )


def _estimate_line_affine(
    scan_band: np.ndarray,
    template_band: np.ndarray,
    config: PixelDiffConfig,
) -> tuple[float, float, float, float] | None:
    height, width = template_band.shape
    _ys, ink_x = np.nonzero(template_band)
    if ink_x.size < 20:
        return None
    active_start = max(0, int(np.percentile(ink_x, 1)))
    active_end = min(width, int(np.percentile(ink_x, 99)) + 1)
    anchors: list[tuple[float, float]] = []
    window = config.line_affine_window_width
    last_start = max(active_start, active_end - window) + 1
    for start in range(
        active_start,
        last_start,
        config.line_affine_window_step,
    ):
        end = min(active_end, start + window)
        if end - start < window // 2:
            continue
        best_shift = 0
        best_iou = _window_iou(scan_band, template_band, start, end, 0)
        for shift in range(-config.line_affine_max_shift, config.line_affine_max_shift + 1):
            score = _window_iou(scan_band, template_band, start, end, shift)
            if score > best_iou:
                best_shift, best_iou = shift, score
        if best_iou >= config.line_affine_min_anchor_iou:
            anchors.append(((start + end) / 2.0, float(-best_shift)))
    if len(anchors) < config.line_affine_min_anchors:
        return None

    xs = np.array([item[0] for item in anchors], dtype=np.float64)
    offsets = np.array([item[1] for item in anchors], dtype=np.float64)
    slope, intercept = np.polyfit(xs, offsets, 1)
    residuals = np.abs(offsets - (slope * xs + intercept))
    median_residual = float(np.median(residuals))
    keep = residuals <= max(3.0, median_residual * 2.5)
    if int(np.count_nonzero(keep)) < config.line_affine_min_anchors:
        return None
    slope, intercept = np.polyfit(xs[keep], offsets[keep], 1)
    if abs(float(slope)) > config.line_affine_max_scale_delta:
        return None

    correction = slope * np.arange(width, dtype=np.float32) + intercept
    if float(np.max(np.abs(correction))) > config.line_affine_max_shift:
        return None
    corrected = _remap_binary(scan_band, correction)
    before_iou = _binary_iou(scan_band, template_band)
    after_iou = _binary_iou(corrected, template_band)
    if after_iou - before_iou < config.line_affine_min_improvement:
        return None
    return float(slope), float(intercept), before_iou, after_iou


def _window_iou(
    scan_band: np.ndarray,
    template_band: np.ndarray,
    start: int,
    end: int,
    shift: int,
) -> float:
    source_start, source_end = start + shift, end + shift
    if source_start < 0 or source_end > scan_band.shape[1]:
        return 0.0
    return _binary_iou(
        scan_band[:, source_start:source_end],
        template_band[:, start:end],
    )


def _remap_binary(image: np.ndarray, correction: np.ndarray) -> np.ndarray:
    height, width = image.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32) - correction.astype(np.float32),
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        image.astype(np.uint8),
        grid_x,
        grid_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def _binary_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return 1.0 if union == 0 else intersection / union


def _foreground(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary.astype(bool)
