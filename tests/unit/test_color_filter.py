from __future__ import annotations

import numpy as np

from pixel_diff.color_filter import remove_colored_marks_bgr
from pixel_diff.models import PixelDiffConfig


def test_remove_colored_marks_whitens_configured_red_and_blue_pixels() -> None:
    image = np.full((3, 3, 3), 255, dtype=np.uint8)
    image[0, 0] = (0, 0, 255)
    image[1, 1] = (255, 0, 0)
    image[2, 2] = (0, 0, 0)

    filtered = remove_colored_marks_bgr(image, PixelDiffConfig())

    assert filtered[0, 0].tolist() == [255, 255, 255]
    assert filtered[1, 1].tolist() == [255, 255, 255]
    assert filtered[2, 2].tolist() == [0, 0, 0]
    assert image[0, 0].tolist() == [0, 0, 255]


def test_remove_colored_marks_can_be_disabled() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = (0, 0, 255)

    filtered = remove_colored_marks_bgr(image, PixelDiffConfig(filter_colored_marks=False))

    assert filtered.tolist() == image.tolist()
    assert filtered is not image


def test_remove_colored_marks_preserves_stable_text_pixels() -> None:
    image = np.full((3, 3, 3), 255, dtype=np.uint8)
    image[0, 0] = (120, 50, 20)
    image[1, 1] = (120, 50, 20)
    stable_text_keep_mask = np.full((3, 3), 255, dtype=np.uint8)
    stable_text_keep_mask[0, 0] = 0

    filtered = remove_colored_marks_bgr(
        image,
        PixelDiffConfig(),
        protected_zero_mask=stable_text_keep_mask,
    )

    assert filtered[0, 0].tolist() == [120, 50, 20]
    assert filtered[1, 1].tolist() == [255, 255, 255]
