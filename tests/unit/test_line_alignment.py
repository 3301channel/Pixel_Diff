from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.line_alignment import (
    _estimate_horizontal_line_offset,
    _filter_monotonic_pairs,
    _match_line_centroids,
    align_text_lines_by_centroid_bgr,
)
from pixel_diff.models import PixelDiffConfig


def test_line_centroid_alignment_moves_shifted_text_line_closer_to_template() -> None:
    template = _page_with_text_line(y=70)
    scan = _page_with_text_line(y=82)
    config = PixelDiffConfig(
        line_centroid_alignment=True,
        line_centroid_max_drift=30,
        line_centroid_row_dilate_width=80,
    )

    result = align_text_lines_by_centroid_bgr(scan, template, config)

    before_delta = abs(_foreground_centroid_y(scan) - _foreground_centroid_y(template))
    after_delta = abs(_foreground_centroid_y(result.aligned_bgr) - _foreground_centroid_y(template))
    assert result.applied
    assert result.matched_pairs == 1
    assert result.max_abs_offset >= 10
    assert after_delta < before_delta


def test_line_centroid_alignment_respects_max_drift_constraint() -> None:
    template = _page_with_text_line(y=70)
    scan = _page_with_text_line(y=120)
    config = PixelDiffConfig(
        line_centroid_alignment=True,
        line_centroid_max_drift=30,
        line_centroid_row_dilate_width=80,
    )

    result = align_text_lines_by_centroid_bgr(scan, template, config)

    assert not result.applied
    assert result.matched_pairs == 0
    assert np.array_equal(result.aligned_bgr, scan)


def test_line_centroid_alignment_does_not_apply_when_offsets_are_zero() -> None:
    template = _page_with_text_line(y=70)
    scan = _page_with_text_line(y=70)
    config = PixelDiffConfig(
        line_centroid_alignment=True,
        line_centroid_max_drift=30,
        line_centroid_row_dilate_width=80,
    )

    result = align_text_lines_by_centroid_bgr(scan, template, config)

    assert not result.applied
    assert result.matched_pairs == 1
    assert result.max_abs_offset == 0.0
    assert np.array_equal(result.aligned_bgr, scan)


def test_line_centroid_matching_can_filter_outlier_offsets_by_median() -> None:
    config = PixelDiffConfig(
        line_centroid_consistency_filter=True,
        line_centroid_median_tolerance=6,
    )

    matched = _match_line_centroids(
        centers_scan=[110, 210, 305, 420],
        centers_template=[100, 200, 300, 400],
        max_drift=30,
        config=config,
    )

    assert matched == [(100, 110), (200, 210), (300, 305)]


def test_line_centroid_matching_requires_monotonic_order_when_enabled() -> None:
    assert _filter_monotonic_pairs([(100, 110), (120, 90), (140, 150)]) == [
        (100, 110),
        (140, 150),
    ]


def test_horizontal_line_offset_estimates_known_shift() -> None:
    template = _page_with_text_line(y=70, x=28)
    scan = _page_with_text_line(y=70, x=34)
    template_foreground = _foreground_mask(template)
    scan_foreground = _foreground_mask(scan)

    estimate = _estimate_horizontal_line_offset(
        scan_foreground,
        template_foreground,
        center_y=60,
        band_half_height=35,
        max_shift=12,
        min_iou=0.40,
        min_improvement=0.02,
    )

    assert estimate is not None
    correction, _best_iou, _improvement = estimate
    assert correction == -6


def test_horizontal_line_alignment_moves_shifted_text_closer_to_template() -> None:
    template = _page_with_text_line(y=70, x=28)
    scan = _page_with_text_line(y=70, x=36)
    config = PixelDiffConfig(
        line_centroid_alignment=True,
        line_centroid_row_dilate_width=80,
        line_horizontal_alignment=True,
        line_horizontal_max_shift=12,
        line_horizontal_min_iou=0.40,
        line_horizontal_min_improvement=0.02,
    )

    result = align_text_lines_by_centroid_bgr(scan, template, config)

    before_delta = abs(_foreground_centroid_x(scan) - _foreground_centroid_x(template))
    after_delta = abs(_foreground_centroid_x(result.aligned_bgr) - _foreground_centroid_x(template))
    assert result.horizontal_applied
    assert result.horizontal_anchors == 1
    assert result.max_abs_horizontal_offset == 8.0
    assert after_delta < before_delta


def test_horizontal_line_alignment_is_disabled_by_default() -> None:
    template = _page_with_text_line(y=70, x=28)
    scan = _page_with_text_line(y=70, x=36)

    result = align_text_lines_by_centroid_bgr(
        scan,
        template,
        PixelDiffConfig(line_centroid_alignment=True, line_centroid_row_dilate_width=80),
    )

    assert not result.horizontal_applied
    assert np.array_equal(result.aligned_bgr, scan)


def _page_with_text_line(y: int, x: int = 28) -> np.ndarray:
    image = np.full((180, 420, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "LINE CENTROID ALIGNMENT",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    return image


def _foreground_centroid_y(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, _ = np.nonzero(binary)
    return float(ys.mean())


def _foreground_centroid_x(image: np.ndarray) -> float:
    _, xs = np.nonzero(_foreground_mask(image))
    return float(xs.mean())


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary
