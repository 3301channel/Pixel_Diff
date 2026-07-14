from __future__ import annotations

import numpy as np
import pytest

from pixel_diff import DimensionMismatchError
from pixel_diff.visualization import (
    GHOST_BACKGROUND_BGR,
    GHOST_OVERLAP_BGR,
    GHOST_SCAN_ONLY_BGR,
    GHOST_TEMPLATE_ONLY_BGR,
    draw_text_ghost_comparison,
)


def test_draw_text_ghost_comparison_colors_binary_stroke_classes() -> None:
    template = np.full((3, 4), 255, dtype=np.uint8)
    scan = np.full((3, 4), 255, dtype=np.uint8)
    template[0, 0] = 0
    scan[0, 1] = 0
    template[1, 2] = 0
    scan[1, 2] = 0

    ghost = draw_text_ghost_comparison(template, scan)

    assert ghost.shape == (3, 4, 3)
    assert tuple(ghost[0, 0]) == GHOST_TEMPLATE_ONLY_BGR
    assert tuple(ghost[0, 1]) == GHOST_SCAN_ONLY_BGR
    assert tuple(ghost[1, 2]) == GHOST_OVERLAP_BGR
    assert tuple(ghost[2, 3]) == GHOST_BACKGROUND_BGR


def test_draw_text_ghost_comparison_requires_matching_grayscale_shapes() -> None:
    with pytest.raises(DimensionMismatchError):
        draw_text_ghost_comparison(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 3), dtype=np.uint8),
        )

    with pytest.raises(DimensionMismatchError):
        draw_text_ghost_comparison(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        )
