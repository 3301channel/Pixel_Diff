from __future__ import annotations

import numpy as np
import pytest

from pixel_diff import DimensionMismatchError
from pixel_diff.models import DifferenceRegion
from pixel_diff.visualization import (
    GHOST_BACKGROUND_BGR,
    GHOST_OVERLAP_BGR,
    GHOST_SCAN_ONLY_BGR,
    GHOST_TEMPLATE_ONLY_BGR,
    _region_label,
    draw_regions,
    draw_text_ghost_comparison,
)


def test_draw_text_ghost_comparison_colors_binary_stroke_classes() -> None:
    assert GHOST_TEMPLATE_ONLY_BGR == (0, 0, 255)
    assert GHOST_SCAN_ONLY_BGR == (255, 255, 0)
    assert GHOST_OVERLAP_BGR == (245, 245, 245)
    assert GHOST_BACKGROUND_BGR == (205, 154, 154)
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


def test_draw_text_ghost_comparison_treats_nearby_strokes_as_overlap() -> None:
    template = np.full((7, 7), 255, dtype=np.uint8)
    scan = np.full((7, 7), 255, dtype=np.uint8)
    template[3, 2] = 0
    scan[3, 4] = 0

    ghost = draw_text_ghost_comparison(template, scan, match_tolerance=2)

    assert tuple(ghost[3, 2]) == GHOST_OVERLAP_BGR
    assert tuple(ghost[3, 4]) == GHOST_OVERLAP_BGR


def test_draw_regions_uses_class_colors_and_chinese_labels() -> None:
    image = np.full((80, 180, 3), 255, dtype=np.uint8)
    regions = [
        DifferenceRegion(1, 5, 25, 30, 30, 900.0, change_type="added", change_label="新增"),
        DifferenceRegion(2, 45, 25, 30, 30, 900.0, change_type="deleted", change_label="删除"),
        DifferenceRegion(3, 85, 25, 30, 30, 900.0, change_type="modified", change_label="修改"),
        DifferenceRegion(
            4,
            125,
            25,
            30,
            30,
            900.0,
            change_type="displaced",
            change_label="文本错位",
        ),
    ]

    output = draw_regions(image, regions, show_classification_labels=True)

    assert tuple(output[55, 5]) == (0, 180, 0)
    assert tuple(output[55, 45]) == (0, 0, 255)
    assert tuple(output[55, 85]) == (0, 165, 255)
    assert tuple(output[55, 125]) == (255, 0, 0)
    assert [_region_label(item, True) for item in regions] == [
        "1 新增", "2 删除", "3 修改", "4 文本错位"
    ]


def test_draw_regions_disabled_labels_restore_red_numeric_style() -> None:
    image = np.full((50, 50, 3), 255, dtype=np.uint8)
    region = DifferenceRegion(
        1, 5, 20, 30, 20, 600.0, change_type="added", change_label="新增"
    )
    output = draw_regions(image, [region], show_classification_labels=False)
    assert tuple(output[20, 5]) == (0, 0, 255)
    assert _region_label(region, False) == "1"
