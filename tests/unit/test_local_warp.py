from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.local_warp import apply_constrained_local_warp_bgr
from pixel_diff.models import PixelDiffConfig


def test_local_warp_returns_original_when_disabled() -> None:
    template = np.full((80, 120, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.rectangle(scan, (20, 30), (80, 50), (0, 0, 0), -1)

    result = apply_constrained_local_warp_bgr(
        scan,
        template,
        PixelDiffConfig(local_warp_enabled=False),
    )

    assert not result.applied
    assert np.array_equal(result.aligned_bgr, scan)
    assert result.max_displacement == 0.0


def test_local_warp_reduces_small_local_translation_residual() -> None:
    template = np.full((160, 220, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.rectangle(template, (48, 58), (168, 90), (0, 0, 0), -1)
    cv2.rectangle(scan, (51, 61), (171, 93), (0, 0, 0), -1)

    before = _mean_abs_gray_delta(scan, template)
    result = apply_constrained_local_warp_bgr(
        scan,
        template,
        PixelDiffConfig(
            local_warp_enabled=True,
            local_warp_max_displacement=6,
            local_warp_scale=0.5,
            local_warp_blur_kernel=9,
        ),
    )
    after = _mean_abs_gray_delta(result.aligned_bgr, template)

    assert result.applied
    assert result.max_displacement <= 6.01
    assert result.mean_displacement > 0.0
    assert after < before


def test_local_warp_gate_skips_identical_pages() -> None:
    image = np.full((120, 180, 3), 255, dtype=np.uint8)
    cv2.putText(image, "same", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    result = apply_constrained_local_warp_bgr(
        image.copy(),
        image,
        PixelDiffConfig(
            local_warp_enabled=True,
            local_warp_gate_enabled=True,
            local_warp_gate_min_iou=0.99,
        ),
    )

    assert not result.applied
    assert result.gate_skipped
    assert result.gate_foreground_iou == 1.0


def _mean_abs_gray_delta(left_bgr: np.ndarray, right_bgr: np.ndarray) -> float:
    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(left, right)))
