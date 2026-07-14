from __future__ import annotations

import numpy as np
import pytest

from pixel_diff import DimensionMismatchError
from pixel_diff.differ import crop_edges, xor_difference


def test_xor_difference_requires_same_shape() -> None:
    with pytest.raises(DimensionMismatchError):
        xor_difference(
            np.zeros((5, 5), dtype=np.uint8),
            np.zeros((5, 6), dtype=np.uint8),
        )


def test_crop_edges_suppresses_border_only() -> None:
    diff = np.full((6, 6), 255, dtype=np.uint8)

    cropped = crop_edges(diff, 2)

    assert cropped[0, 3] == 0
    assert cropped[5, 3] == 0
    assert cropped[3, 0] == 0
    assert cropped[3, 5] == 0
    assert cropped[3, 3] == 255
    assert diff[0, 0] == 255
