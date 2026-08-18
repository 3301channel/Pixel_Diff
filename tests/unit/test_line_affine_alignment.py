from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.line_affine_alignment import align_text_lines_affine_bgr
from pixel_diff.models import PixelDiffConfig


def test_line_affine_alignment_corrects_scale_drift_and_keeps_real_change() -> None:
    template = np.full((120, 900, 3), 255, dtype=np.uint8)
    cv2.putText(
        template,
        "AFFINE TEXT ALIGNMENT SAMPLE 0123456789",
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    matrix = np.array([[1.025, 0.0, -5.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    scan = cv2.warpAffine(
        template,
        matrix,
        (template.shape[1], template.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    cv2.rectangle(scan, (815, 42), (824, 68), (0, 0, 0), -1)
    config = PixelDiffConfig(
        line_affine_alignment_enabled=True,
        line_affine_window_width=220,
        line_affine_window_step=140,
        line_affine_max_shift=35,
        line_affine_min_anchors=3,
        line_affine_min_anchor_iou=0.15,
        line_affine_min_improvement=0.02,
        line_affine_max_scale_delta=0.04,
        line_affine_band_half_height=32,
    )

    result = align_text_lines_affine_bgr(scan, template, [60], config)

    assert result.applied_lines == 1
    assert result.after_iou > result.before_iou + 0.10
    changed_roi = result.aligned_bgr[42:69, 795:815]
    assert np.count_nonzero(cv2.cvtColor(changed_roi, cv2.COLOR_BGR2GRAY) < 80) > 30


def test_line_affine_alignment_is_noop_when_disabled() -> None:
    image = np.full((40, 80, 3), 255, dtype=np.uint8)

    result = align_text_lines_affine_bgr(
        image,
        image,
        [20],
        PixelDiffConfig(line_affine_alignment_enabled=False),
    )

    assert result.applied_lines == 0
    assert result.aligned_bgr is image

