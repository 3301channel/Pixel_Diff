"""Rigid integer translation of complete text foreground blocks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RigidBlockMove:
    source_start: int
    source_end: int
    dx: int
    confidence: float


@dataclass(frozen=True)
class RigidBlockAlignmentResult:
    aligned_bgr: np.ndarray
    attempted: int
    applied: int
    rejected_overlap: int
    rejected_quality: int
    before_iou: float
    after_iou: float


def segment_rigid_text_blocks(
    scan_foreground: np.ndarray,
    template_foreground: np.ndarray,
    min_gap_width: int,
    max_internal_gap: int,
) -> list[tuple[int, int]]:
    """Return occupied horizontal blocks split only at stable whitespace runs."""

    occupied = np.any(scan_foreground | template_foreground, axis=0)
    components = _occupied_runs(occupied)
    if not components:
        return []
    blocks = [components[0]]
    for start, end in components[1:]:
        previous_start, previous_end = blocks[-1]
        gap = start - previous_end
        if gap < min_gap_width or gap <= max_internal_gap:
            blocks[-1] = (previous_start, end)
        else:
            blocks.append((start, end))
    return blocks


def apply_rigid_block_moves_bgr(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    moves: list[RigidBlockMove],
    min_iou_improvement: float,
) -> RigidBlockAlignmentResult:
    """Move complete foreground blocks using exact integer pixel copies."""

    scan_fg = _foreground(scan_bgr)
    template_fg = _foreground(template_bgr)
    output = scan_bgr.copy()
    occupied = scan_fg.copy()
    applied = 0
    rejected_overlap = 0
    height, width = scan_fg.shape
    for move in moves:
        target_start = move.source_start + move.dx
        target_end = move.source_end + move.dx
        if (
            move.source_start < 0
            or move.source_end > width
            or move.source_end <= move.source_start
            or target_start < 0
            or target_end > width
        ):
            rejected_overlap += 1
            continue
        source_mask = scan_fg[:, move.source_start : move.source_end]
        if not np.any(source_mask):
            continue
        collision_map = occupied.copy()
        collision_map[:, move.source_start : move.source_end] &= ~source_mask
        if np.any(collision_map[:, target_start:target_end] & source_mask):
            rejected_overlap += 1
            continue

        source_pixels = scan_bgr[:, move.source_start : move.source_end].copy()
        output[:, move.source_start : move.source_end][source_mask] = 255
        target_view = output[:, target_start:target_end]
        target_view[source_mask] = source_pixels[source_mask]
        occupied[:, move.source_start : move.source_end] &= ~source_mask
        occupied[:, target_start:target_end] |= source_mask
        applied += 1

    before_iou = _iou(scan_fg, template_fg)
    output_fg = _foreground(output)
    after_iou = _iou(output_fg, template_fg)
    rejected_quality = 0
    if applied and after_iou - before_iou < min_iou_improvement:
        rejected_quality = applied
        output = scan_bgr
        applied = 0
        after_iou = before_iou
    return RigidBlockAlignmentResult(
        aligned_bgr=output,
        attempted=len(moves),
        applied=applied,
        rejected_overlap=rejected_overlap,
        rejected_quality=rejected_quality,
        before_iou=before_iou,
        after_iou=after_iou,
    )


def _occupied_runs(occupied: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.append(occupied, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _foreground(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _threshold, binary = cv2.threshold(
        gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return binary.astype(bool)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    return 1.0 if union == 0 else int(np.count_nonzero(left & right)) / union
