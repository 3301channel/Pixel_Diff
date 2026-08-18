"""Confidence-gated second-pass horizontal alignment for text residuals."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pixel_diff.models import PixelDiffConfig


@dataclass(frozen=True)
class ResidualLineAlignmentResult:
    aligned_bgr: np.ndarray
    scan_binary: np.ndarray
    diff_mask: np.ndarray
    candidate_lines: int = 0
    attempted_lines: int = 0
    applied_lines: int = 0
    before_diff_pixels: int = 0
    after_diff_pixels: int = 0
    before_long_residuals: int = 0
    after_long_residuals: int = 0
    protected_intervals: int = 0
    protected_retention: float = 1.0
    max_displacement: float = 0.0
    max_scale_delta: float = 0.0


def realign_residual_text_lines_bgr(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    scan_binary: np.ndarray,
    template_binary: np.ndarray,
    diff_mask: np.ndarray,
    line_centers: list[int],
    config: PixelDiffConfig,
) -> ResidualLineAlignmentResult:
    """Realign only residual-heavy lines when every safety gate improves."""
    before_pixels = int(np.count_nonzero(diff_mask))
    if not config.residual_line_realignment_enabled or not line_centers:
        return ResidualLineAlignmentResult(
            scan_bgr,
            scan_binary,
            diff_mask,
            before_diff_pixels=before_pixels,
            after_diff_pixels=before_pixels,
        )

    height = diff_mask.shape[0]
    half_height = config.line_affine_band_half_height
    candidates = 0
    attempted = 0
    applied = 0
    before_long = 0
    after_long = 0
    protected_count = 0
    protected_retentions: list[float] = []
    max_displacement = 0.0
    max_scale = 0.0
    output_bgr = scan_bgr.copy()
    output_binary = scan_binary.copy()
    output_diff = diff_mask.copy()
    added = cv2.bitwise_and(scan_binary, cv2.bitwise_not(template_binary))
    deleted = cv2.bitwise_and(template_binary, cv2.bitwise_not(scan_binary))
    for center in line_centers:
        y0 = max(0, center - half_height)
        y1 = min(height, center + half_height + 1)
        is_candidate, _ = _is_candidate_band(
            added[y0:y1], deleted[y0:y1], config
        )
        if is_candidate:
            candidates += 1
            attempted += 1
            _, band_before_area = _long_residual_stats(
                output_diff[y0:y1], config.residual_line_min_span
            )
            before_long += band_before_area
            estimate = _estimate_residual_offsets(
                output_binary[y0:y1] > 0,
                template_binary[y0:y1] > 0,
                config,
            )
            if estimate is None:
                after_long += band_before_area
                continue
            offsets, protected, scale_delta = estimate
            corrected_binary = _remap_mask(output_binary[y0:y1], offsets)
            corrected_bgr = _remap_bgr(output_bgr[y0:y1], offsets)
            for start, end in protected:
                corrected_binary[:, start:end] = output_binary[y0:y1, start:end]
                corrected_bgr[:, start:end] = output_bgr[y0:y1, start:end]
            corrected_diff = cv2.bitwise_xor(
                corrected_binary, template_binary[y0:y1]
            )
            accepted, retention, corrected_residual_area = _accept_band(
                output_binary[y0:y1],
                corrected_binary,
                template_binary[y0:y1],
                output_diff[y0:y1],
                corrected_diff,
                protected,
                config,
            )
            if not accepted:
                after_long += band_before_area
                continue
            output_bgr[y0:y1] = corrected_bgr
            output_binary[y0:y1] = corrected_binary
            output_diff[y0:y1] = corrected_diff
            applied += 1
            after_long += corrected_residual_area
            protected_count += len(protected)
            protected_retentions.append(retention)
            max_displacement = max(
                max_displacement, float(np.max(np.abs(offsets)))
            )
            max_scale = max(max_scale, scale_delta)

    return ResidualLineAlignmentResult(
        output_bgr,
        output_binary,
        output_diff,
        candidate_lines=candidates,
        attempted_lines=attempted,
        applied_lines=applied,
        before_diff_pixels=before_pixels,
        after_diff_pixels=int(np.count_nonzero(output_diff)),
        before_long_residuals=before_long,
        after_long_residuals=after_long,
        protected_intervals=protected_count,
        protected_retention=min(protected_retentions, default=1.0),
        max_displacement=max_displacement,
        max_scale_delta=max_scale,
    )


def _estimate_residual_offsets(
    scan: np.ndarray,
    template: np.ndarray,
    config: PixelDiffConfig,
) -> tuple[np.ndarray, list[tuple[int, int]], float] | None:
    width = template.shape[1]
    window = min(config.residual_line_window_width, width)
    anchors: list[tuple[int, int]] = []
    previous_source = -1
    for start in range(0, max(1, width - window + 1), config.residual_line_window_step):
        end = min(width, start + window)
        if np.count_nonzero(template[:, start:end]) < 10:
            continue
        best_shift, best_score = 0, -1.0
        for shift in range(-config.residual_line_max_shift, config.residual_line_max_shift + 1):
            source_start, source_end = start + shift, end + shift
            if source_start < 0 or source_end > width:
                continue
            score = _projection_similarity(
                scan[:, source_start:source_end], template[:, start:end]
            )
            if score > best_score:
                best_shift, best_score = shift, score
        center = (start + end) // 2
        source_center = center + best_shift
        if (
            best_score >= config.residual_line_min_anchor_similarity
            and source_center > previous_source
        ):
            anchors.append((center, best_shift))
            previous_source = source_center
    if len(anchors) < config.residual_line_min_anchors:
        return None

    offsets = np.zeros(width, dtype=np.float32)
    protected: list[tuple[int, int]] = []
    centers = [anchor[0] for anchor in anchors]
    shifts = [anchor[1] for anchor in anchors]
    offsets[: centers[0]] = shifts[0]
    offsets[centers[-1] :] = shifts[-1]
    max_scale = 0.0
    for index in range(len(anchors) - 1):
        x0, shift0 = anchors[index]
        x1, shift1 = anchors[index + 1]
        delta = shift1 - shift0
        scale_delta = abs(delta / max(1, x1 - x0))
        max_scale = max(max_scale, scale_delta)
        if abs(delta) >= config.residual_line_jump_threshold:
            midpoint = (x0 + x1) // 2
            offsets[x0:midpoint] = shift0
            offsets[midpoint:x1] = shift1
            half = config.residual_line_protection_width // 2
            protected.append((max(0, midpoint - half), min(width, midpoint + half + 1)))
        else:
            if scale_delta > config.residual_line_max_scale_delta:
                return None
            offsets[x0:x1] = np.linspace(shift0, shift1, x1 - x0, endpoint=False)
    if float(np.max(np.abs(offsets))) > config.residual_line_max_shift:
        return None
    return offsets, protected, max_scale


def _accept_band(
    before_scan: np.ndarray,
    after_scan: np.ndarray,
    template: np.ndarray,
    before_diff: np.ndarray,
    after_diff: np.ndarray,
    protected: list[tuple[int, int]],
    config: PixelDiffConfig,
) -> tuple[bool, float, int]:
    before_pixels = int(np.count_nonzero(before_diff))
    after_pixels = int(np.count_nonzero(after_diff))
    reduction = (before_pixels - after_pixels) / max(1, before_pixels)
    iou_gain = _iou(after_scan > 0, template > 0) - _iou(before_scan > 0, template > 0)
    protected_before = 0
    protected_after = 0
    for start, end in protected:
        protected_before += int(np.count_nonzero(before_diff[:, start:end]))
        protected_after += int(np.count_nonzero(after_diff[:, start:end]))
    retention = (
        protected_after / protected_before if protected_before else 1.0
    )
    before_long_count, before_long_area = _long_residual_stats(
        before_diff, config.residual_line_min_span
    )
    after_long_count, after_long_area = _long_residual_stats(
        after_diff, config.residual_line_min_span
    )
    accepted = (
        iou_gain >= config.residual_line_min_iou_improvement
        and reduction >= config.residual_line_min_diff_reduction
        and after_long_count <= before_long_count
        and after_long_area < before_long_area
        and retention >= config.residual_line_min_protected_retention
    )
    return accepted, retention, after_long_area


def _long_residual_stats(mask: np.ndarray, min_span: int) -> tuple[int, int]:
    kernel_width = max(3, min(31, max(3, min_span // 8)))
    connected = cv2.morphologyEx(
        (mask > 0).astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((1, kernel_width), dtype=np.uint8),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    width_threshold = max(8, min_span // 2)
    long_stats = [stats[index] for index in range(1, count) if stats[index, 2] >= width_threshold]
    return len(long_stats), sum(int(item[4]) for item in long_stats)


def _remap_mask(mask: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32) + offsets,
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        mask, grid_x, grid_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _remap_bgr(image: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32) + offsets,
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        image, grid_x, grid_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    return 1.0 if union == 0 else int(np.count_nonzero(left & right)) / union


def _projection_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_projection = np.count_nonzero(left, axis=0).astype(np.float32)
    right_projection = np.count_nonzero(right, axis=0).astype(np.float32)
    denominator = float(
        np.linalg.norm(left_projection) * np.linalg.norm(right_projection)
    )
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_projection, right_projection) / denominator)


def _is_candidate_band(
    added: np.ndarray,
    deleted: np.ndarray,
    config: PixelDiffConfig,
) -> tuple[bool, int]:
    added_boxes = _component_boxes(added)
    deleted_boxes = _component_boxes(deleted)
    boxes = added_boxes + deleted_boxes
    if len(boxes) < config.residual_line_min_components:
        return False, len(boxes)
    x0 = min(box[0] for box in boxes)
    x1 = max(box[0] + box[2] for box in boxes)
    if x1 - x0 < config.residual_line_min_span:
        return False, len(boxes)
    if not added_boxes or not deleted_boxes:
        return False, len(boxes)
    added_y0 = min(box[1] for box in added_boxes)
    added_y1 = max(box[1] + box[3] for box in added_boxes)
    deleted_y0 = min(box[1] for box in deleted_boxes)
    deleted_y1 = max(box[1] + box[3] for box in deleted_boxes)
    vertical_overlap = min(added_y1, deleted_y1) - max(added_y0, deleted_y0)
    return vertical_overlap > 0, len(boxes)


def _component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    return [
        (
            int(stats[index, 0]),
            int(stats[index, 1]),
            int(stats[index, 2]),
            int(stats[index, 3]),
        )
        for index in range(1, count)
    ]
