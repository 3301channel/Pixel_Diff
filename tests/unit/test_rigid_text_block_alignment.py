from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.rigid_text_block_alignment import (
    RigidBlockMove,
    apply_rigid_block_moves_bgr,
    segment_rigid_text_blocks,
)


def _to_bgr(foreground: np.ndarray) -> np.ndarray:
    gray = np.where(foreground, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_segmentation_keeps_small_english_gaps_inside_one_block() -> None:
    foreground = np.zeros((30, 100), dtype=bool)
    foreground[5:25, 8:18] = True
    foreground[5:25, 22:32] = True
    foreground[5:25, 45:57] = True

    blocks = segment_rigid_text_blocks(
        foreground,
        foreground,
        min_gap_width=4,
        max_internal_gap=10,
    )

    assert blocks == [(8, 32), (45, 57)]


def test_integer_move_preserves_glyph_shape_and_removes_source_strokes() -> None:
    scan = np.zeros((36, 120), dtype=bool)
    scan[6:30, 20:32] = True
    scan[15:20, 20:42] = True
    template = np.zeros_like(scan)
    template[:, 14:36] = scan[:, 20:42]

    result = apply_rigid_block_moves_bgr(
        _to_bgr(scan),
        _to_bgr(template),
        [RigidBlockMove(source_start=20, source_end=42, dx=-6, confidence=0.95)],
        min_iou_improvement=0.01,
    )
    corrected = cv2.cvtColor(result.aligned_bgr, cv2.COLOR_BGR2GRAY) < 128
    source_component = scan[:, 20:42]
    moved_component = corrected[:, 14:36]

    assert result.applied == 1
    assert np.array_equal(source_component, moved_component)
    assert np.count_nonzero(corrected[:, 36:42]) == 0
    assert cv2.connectedComponents(scan.astype(np.uint8))[0] == cv2.connectedComponents(
        corrected.astype(np.uint8)
    )[0]


def test_destination_collision_rejects_move_and_preserves_input() -> None:
    scan = np.zeros((30, 100), dtype=bool)
    scan[5:25, 15:25] = True
    scan[5:25, 35:45] = True

    result = apply_rigid_block_moves_bgr(
        _to_bgr(scan),
        _to_bgr(scan),
        [RigidBlockMove(source_start=15, source_end=25, dx=20, confidence=0.95)],
        min_iou_improvement=0.0,
    )

    assert result.applied == 0
    assert result.rejected_overlap == 1
    assert np.array_equal(result.aligned_bgr, _to_bgr(scan))
