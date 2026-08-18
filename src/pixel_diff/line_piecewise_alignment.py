"""Confidence-gated monotonic piecewise horizontal alignment for text lines."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pixel_diff.models import PixelDiffConfig
from pixel_diff.rigid_text_block_alignment import (
    RigidBlockMove,
    apply_rigid_block_moves_bgr,
    segment_rigid_text_blocks,
)


@dataclass(frozen=True)
class LinePiecewiseAlignmentResult:
    aligned_bgr: np.ndarray
    applied_lines: int
    checked_lines: int
    anchors: int
    protected_intervals: int
    before_iou: float
    after_iou: float
    max_displacement: float
    max_scale_delta: float
    rigid_blocks_attempted: int = 0
    rigid_blocks_applied: int = 0
    rigid_blocks_rejected_overlap: int = 0
    rigid_blocks_rejected_quality: int = 0
    rigid_block_applied_lines: int = 0


def align_text_lines_piecewise_bgr(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    line_centers: list[int],
    config: PixelDiffConfig,
) -> LinePiecewiseAlignmentResult:
    if not config.line_piecewise_alignment_enabled or not line_centers:
        return LinePiecewiseAlignmentResult(scan_bgr, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
    scan_fg = _foreground(scan_bgr)
    template_fg = _foreground(template_bgr)
    output = scan_bgr.copy()
    before_scores: list[float] = []
    after_scores: list[float] = []
    total_anchors = 0
    total_protected = 0
    max_displacement = 0.0
    max_scale = 0.0
    applied = 0
    rigid_attempted = 0
    rigid_applied = 0
    rigid_rejected_overlap = 0
    rigid_rejected_quality = 0
    rigid_applied_lines = 0
    height, width = scan_fg.shape
    half_height = config.line_affine_band_half_height
    for center in line_centers:
        y0, y1 = max(0, center - half_height), min(height, center + half_height + 1)
        anchors = _estimate_anchors(scan_fg[y0:y1], template_fg[y0:y1], config)
        has_large_jump = any(
            abs(anchors[index + 1][1] - anchors[index][1])
            > config.line_piecewise_jump_threshold
            for index in range(len(anchors) - 1)
        )
        if config.rigid_text_block_alignment_enabled and has_large_jump:
            blocks = segment_rigid_text_blocks(
                scan_fg[y0:y1],
                template_fg[y0:y1],
                config.rigid_text_block_min_gap_width,
                config.rigid_text_block_max_internal_gap,
            )
            moves = _rigid_moves_from_anchors(blocks, anchors, config)
            rigid_result = apply_rigid_block_moves_bgr(
                scan_bgr[y0:y1],
                template_bgr[y0:y1],
                moves,
                config.rigid_text_block_min_iou_improvement,
            )
            rigid_attempted += rigid_result.attempted
            rigid_applied += rigid_result.applied
            rigid_rejected_overlap += rigid_result.rejected_overlap
            rigid_rejected_quality += rigid_result.rejected_quality
            if rigid_result.applied:
                output[y0:y1] = rigid_result.aligned_bgr
                applied += 1
                rigid_applied_lines += 1
                before_scores.append(rigid_result.before_iou)
                after_scores.append(rigid_result.after_iou)
            continue
        result = _align_band(scan_fg[y0:y1], template_fg[y0:y1], config)
        if result is None:
            continue
        offsets, protected, before_iou, after_iou, anchor_count, scale_delta = result
        grid_x, grid_y = np.meshgrid(
            np.arange(width, dtype=np.float32) + offsets,
            np.arange(y1 - y0, dtype=np.float32),
        )
        corrected = cv2.remap(
            scan_bgr[y0:y1], grid_x, grid_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        for start, end in protected:
            corrected[:, start:end] = scan_bgr[y0:y1, start:end]
        output[y0:y1] = corrected
        applied += 1
        total_anchors += anchor_count
        total_protected += len(protected)
        before_scores.append(before_iou)
        after_scores.append(after_iou)
        max_displacement = max(max_displacement, float(np.max(np.abs(offsets))))
        max_scale = max(max_scale, scale_delta)
    return LinePiecewiseAlignmentResult(
        output, applied, len(line_centers), total_anchors, total_protected,
        float(np.mean(before_scores)) if before_scores else 0.0,
        float(np.mean(after_scores)) if after_scores else 0.0,
        max_displacement, max_scale, rigid_attempted, rigid_applied,
        rigid_rejected_overlap, rigid_rejected_quality, rigid_applied_lines,
    )


def _rigid_moves_from_anchors(
    blocks: list[tuple[int, int]],
    anchors: list[tuple[int, int, float]],
    config: PixelDiffConfig,
) -> list[RigidBlockMove]:
    moves: list[RigidBlockMove] = []
    for start, end in blocks:
        center = (start + end) / 2.0
        nearest = min(anchors, key=lambda anchor: abs(anchor[0] - center))
        _anchor_x, shift, confidence = nearest
        if confidence < config.rigid_text_block_min_anchor_similarity or shift == 0:
            continue
        moves.append(RigidBlockMove(start, end, -int(shift), float(confidence)))
    return moves


def _align_band(
    scan: np.ndarray, template: np.ndarray, config: PixelDiffConfig
) -> tuple[np.ndarray, list[tuple[int, int]], float, float, int, float] | None:
    width = template.shape[1]
    anchors = _estimate_anchors(scan, template, config)
    if len(anchors) < config.line_piecewise_min_anchors:
        return None

    offsets = np.zeros(width, dtype=np.float32)
    protected: list[tuple[int, int]] = []
    max_scale = 0.0
    centers = [item[0] for item in anchors]
    shifts = [item[1] for item in anchors]
    offsets[: centers[0]] = shifts[0]
    offsets[centers[-1] :] = shifts[-1]
    for index in range(len(anchors) - 1):
        x0, shift0, _ = anchors[index]
        x1, shift1, _ = anchors[index + 1]
        delta = shift1 - shift0
        scale_delta = abs(delta / max(1, x1 - x0))
        max_scale = max(max_scale, scale_delta)
        if abs(delta) > config.line_piecewise_jump_threshold:
            midpoint = (x0 + x1) // 2
            offsets[x0:midpoint] = shift0
            offsets[midpoint:x1] = shift1
            half = config.line_piecewise_protection_width // 2
            protected.append((max(0, midpoint - half), min(width, midpoint + half + 1)))
        else:
            if scale_delta > config.line_piecewise_max_scale_delta:
                return None
            offsets[x0:x1] = np.linspace(shift0, shift1, x1 - x0, endpoint=False)
    corrected = _remap(scan, offsets)
    for start, end in protected:
        corrected[:, start:end] = scan[:, start:end]
    before_iou, after_iou = _iou(scan, template), _iou(corrected, template)
    if after_iou - before_iou < config.line_piecewise_min_improvement:
        return None
    return offsets, protected, before_iou, after_iou, len(anchors), max_scale


def _estimate_anchors(
    scan: np.ndarray,
    template: np.ndarray,
    config: PixelDiffConfig,
) -> list[tuple[int, int, float]]:
    width = template.shape[1]
    window = config.line_piecewise_window_width
    anchors: list[tuple[int, int, float]] = []
    previous_source = -1
    for start in range(0, max(1, width - window + 1), config.line_piecewise_window_step):
        end = min(width, start + window)
        if np.count_nonzero(template[:, start:end]) < 10:
            continue
        best_shift, best_score = 0, -1.0
        for shift in range(-config.line_piecewise_max_shift, config.line_piecewise_max_shift + 1):
            source_start, source_end = start + shift, end + shift
            if source_start < 0 or source_end > width:
                continue
            score = _iou(scan[:, source_start:source_end], template[:, start:end])
            if score > best_score:
                best_shift, best_score = shift, score
        center = (start + end) // 2
        source_center = center + best_shift
        if (
            best_score >= config.line_piecewise_min_anchor_similarity
            and source_center > previous_source
        ):
            anchors.append((center, best_shift, best_score))
            previous_source = source_center
    return anchors


def _remap(image: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    height, width = image.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32) + offsets,
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        image.astype(np.uint8), grid_x, grid_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    return 1.0 if union == 0 else int(np.count_nonzero(left & right)) / union


def _foreground(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary.astype(bool)
