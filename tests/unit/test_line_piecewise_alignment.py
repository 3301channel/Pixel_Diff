from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.line_piecewise_alignment import align_text_lines_piecewise_bgr
from pixel_diff.models import PixelDiffConfig


def _to_bgr(foreground: np.ndarray) -> np.ndarray:
    gray = np.where(foreground, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _text_line() -> np.ndarray:
    line = np.zeros((48, 240), dtype=bool)
    for index, x in enumerate(range(15, 195, 30)):
        height = 12 + index * 3
        line[8 : 8 + height, x : x + 5 + index] = True
        line[28:32, x : x + 12] = True
    return line


def test_piecewise_alignment_realigns_suffix_after_inserted_gap() -> None:
    template = _text_line()
    scan = np.zeros_like(template)
    split = 100
    shift = 12
    scan[:, :split] = template[:, :split]
    scan[:, split + shift :] = template[:, split:-shift]
    scan[12:34, split + 2 : split + 8] = True
    config = PixelDiffConfig(
        line_piecewise_alignment_enabled=True,
        line_piecewise_window_width=48,
        line_piecewise_window_step=24,
        line_piecewise_max_shift=20,
        line_piecewise_min_anchor_similarity=0.45,
        line_piecewise_min_anchors=3,
        line_piecewise_jump_threshold=6,
        line_piecewise_protection_width=18,
        line_piecewise_min_improvement=0.01,
    )

    result = align_text_lines_piecewise_bgr(
        _to_bgr(scan), _to_bgr(template), [24], config
    )
    corrected = cv2.cvtColor(result.aligned_bgr, cv2.COLOR_BGR2GRAY) < 128

    assert result.applied_lines == 1
    before_suffix = np.count_nonzero(scan[:, 135:] ^ template[:, 135:])
    after_suffix = np.count_nonzero(corrected[:, 135:] ^ template[:, 135:])
    assert after_suffix < before_suffix * 0.35
    assert np.count_nonzero(corrected[:, 96:122] ^ template[:, 96:122]) > 0
    assert result.protected_intervals >= 1


def test_piecewise_alignment_rolls_back_unrelated_lines() -> None:
    template = _text_line()
    scan = np.zeros_like(template)
    cv2.circle(scan.astype(np.uint8), (120, 24), 15, 1, -1)

    result = align_text_lines_piecewise_bgr(
        _to_bgr(scan),
        _to_bgr(template),
        [24],
        PixelDiffConfig(line_piecewise_alignment_enabled=True),
    )

    assert result.applied_lines == 0
    assert np.array_equal(result.aligned_bgr, _to_bgr(scan))


def test_large_jump_uses_rigid_blocks_without_changing_foreground_count() -> None:
    template = _text_line()
    scan = np.zeros_like(template)
    split = 100
    shift = 12
    scan[:, :split] = template[:, :split]
    scan[:, split + shift :] = template[:, split:-shift]
    scan[12:34, split + 2 : split + 8] = True
    config = PixelDiffConfig(
        line_piecewise_alignment_enabled=True,
        line_piecewise_window_width=48,
        line_piecewise_window_step=24,
        line_piecewise_max_shift=20,
        line_piecewise_min_anchor_similarity=0.45,
        line_piecewise_min_anchors=3,
        line_piecewise_jump_threshold=6,
        line_piecewise_min_improvement=0.01,
        rigid_text_block_alignment_enabled=True,
        rigid_text_block_min_gap_width=4,
        rigid_text_block_max_internal_gap=10,
        rigid_text_block_min_anchor_similarity=0.45,
        rigid_text_block_min_iou_improvement=0.01,
    )

    result = align_text_lines_piecewise_bgr(
        _to_bgr(scan), _to_bgr(template), [24], config
    )
    corrected = cv2.cvtColor(result.aligned_bgr, cv2.COLOR_BGR2GRAY) < 128

    assert result.rigid_blocks_applied > 0
    assert np.count_nonzero(corrected) == np.count_nonzero(scan)
    assert cv2.connectedComponents(corrected.astype(np.uint8))[0] == cv2.connectedComponents(
        scan.astype(np.uint8)
    )[0]
