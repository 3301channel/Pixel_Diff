from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.models import PixelDiffConfig
from pixel_diff.residual_line_alignment import realign_residual_text_lines_bgr


def _to_bgr(foreground: np.ndarray) -> np.ndarray:
    gray = np.where(foreground, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _binary(foreground: np.ndarray) -> np.ndarray:
    return np.where(foreground, 255, 0).astype(np.uint8)


def _text_line() -> np.ndarray:
    line = np.zeros((48, 240), dtype=bool)
    for index, x in enumerate(range(15, 210, 28)):
        width = 6 + index % 3
        line[8:34, x : x + width] = True
        line[18:23, x : x + 15] = True
    return line


def test_disabled_residual_realigner_is_exact_noop() -> None:
    template = _text_line()
    scan = np.roll(template, 3, axis=1)
    scan_binary = _binary(scan)
    template_binary = _binary(template)
    diff = cv2.bitwise_xor(scan_binary, template_binary)
    scan_bgr = _to_bgr(scan)

    result = realign_residual_text_lines_bgr(
        scan_bgr,
        _to_bgr(template),
        scan_binary,
        template_binary,
        diff,
        [24],
        PixelDiffConfig(),
    )

    assert result.applied_lines == 0
    assert np.array_equal(result.aligned_bgr, scan_bgr)
    assert np.array_equal(result.scan_binary, scan_binary)
    assert np.array_equal(result.diff_mask, diff)


def test_isolated_character_difference_is_not_a_candidate() -> None:
    template = _text_line()
    scan = template.copy()
    scan[10:33, 116:124] = True
    scan_binary = _binary(scan)
    template_binary = _binary(template)
    diff = cv2.bitwise_xor(scan_binary, template_binary)

    result = realign_residual_text_lines_bgr(
        _to_bgr(scan),
        _to_bgr(template),
        scan_binary,
        template_binary,
        diff,
        [24],
        PixelDiffConfig(
            residual_line_realignment_enabled=True,
            residual_line_min_span=80,
            residual_line_min_components=3,
        ),
    )

    assert result.candidate_lines == 0
    assert result.applied_lines == 0


def test_realigns_stretched_suffix_but_preserves_inserted_character() -> None:
    template = _text_line()
    scan = np.zeros_like(template)
    split = 92
    shift = 12
    scan[:, :split] = template[:, :split]
    scan[:, split + shift :] = template[:, split:-shift]
    scan[10:35, split + 2 : split + 8] = True
    scan_binary = _binary(scan)
    template_binary = _binary(template)
    diff = cv2.bitwise_xor(scan_binary, template_binary)
    config = PixelDiffConfig(
        residual_line_realignment_enabled=True,
        residual_line_window_width=44,
        residual_line_window_step=18,
        residual_line_max_shift=18,
        residual_line_min_anchor_similarity=0.40,
        residual_line_min_anchors=3,
        residual_line_min_span=70,
        residual_line_min_components=3,
        residual_line_jump_threshold=5,
        residual_line_protection_width=18,
        residual_line_max_scale_delta=0.25,
        residual_line_min_iou_improvement=0.005,
        residual_line_min_diff_reduction=0.05,
        residual_line_min_protected_retention=0.60,
    )

    result = realign_residual_text_lines_bgr(
        _to_bgr(scan),
        _to_bgr(template),
        scan_binary,
        template_binary,
        diff,
        [24],
        config,
    )

    assert result.candidate_lines == 1
    assert result.applied_lines == 1
    assert result.after_diff_pixels < result.before_diff_pixels
    assert result.after_long_residuals < result.before_long_residuals
    assert result.protected_intervals >= 1
    assert result.protected_retention >= config.residual_line_min_protected_retention


def test_projection_anchors_tolerate_small_vertical_scan_jitter() -> None:
    template = _text_line()
    scan = np.zeros_like(template)
    shifted = np.roll(template, 3, axis=0)
    scan[:, :92] = shifted[:, :92]
    scan[:, 104:] = shifted[:, 92:-12]
    scan[13:36, 95:101] = True
    scan_binary = _binary(scan)
    template_binary = _binary(template)
    diff = cv2.bitwise_xor(scan_binary, template_binary)

    result = realign_residual_text_lines_bgr(
        _to_bgr(scan),
        _to_bgr(template),
        scan_binary,
        template_binary,
        diff,
        [24],
        PixelDiffConfig(
            residual_line_realignment_enabled=True,
            residual_line_window_width=44,
            residual_line_window_step=18,
            residual_line_max_shift=18,
            residual_line_min_anchor_similarity=0.70,
            residual_line_min_anchors=3,
            residual_line_min_span=70,
            residual_line_min_components=3,
            residual_line_jump_threshold=5,
            residual_line_protection_width=18,
            residual_line_max_scale_delta=0.25,
            residual_line_min_iou_improvement=0.0,
            residual_line_min_diff_reduction=0.01,
            residual_line_min_protected_retention=0.60,
        ),
    )

    assert result.applied_lines == 1
