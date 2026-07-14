from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.models import DifferenceRegion, PixelDiffConfig
from pixel_diff.regions import extract_regions, filter_locally_similar_regions


def test_extract_regions_filters_numbers_and_sorts_stably() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (50, 10), (70, 20), 255, -1)
    cv2.rectangle(mask, (10, 10), (20, 20), 255, -1)
    cv2.rectangle(mask, (30, 40), (31, 41), 255, -1)

    regions = extract_regions(mask, min_area=20)

    assert [region.id for region in regions] == [1, 2]
    assert [(region.x, region.y) for region in regions] == [(10, 10), (50, 10)]
    assert all(region.width > 0 and region.height > 0 for region in regions)


def test_filter_locally_similar_regions_removes_shift_residuals() -> None:
    template = np.full((40, 40), 255, dtype=np.uint8)
    scan = np.full((40, 40), 255, dtype=np.uint8)
    cv2.rectangle(template, (10, 10), (18, 18), 0, -1)
    cv2.rectangle(scan, (12, 10), (20, 18), 0, -1)
    regions = [
        DifferenceRegion(id=7, x=9, y=9, width=13, height=11, area=120.0),
    ]
    config = PixelDiffConfig(
        local_similarity_iou_threshold=0.80,
        local_similarity_padding=2,
        local_similarity_search_radius=3,
        sparse_residual_max_area=0,
        small_residual_max_area=0,
    )

    filtered = filter_locally_similar_regions(regions, scan, template, config)

    assert filtered == []


def test_filter_locally_similar_regions_keeps_unmatched_change_and_renumbers() -> None:
    template = np.full((40, 40), 255, dtype=np.uint8)
    scan = np.full((40, 40), 255, dtype=np.uint8)
    cv2.rectangle(template, (10, 10), (18, 18), 0, -1)
    cv2.rectangle(scan, (26, 10), (34, 18), 0, -1)
    regions = [
        DifferenceRegion(id=7, x=25, y=9, width=11, height=11, area=120.0),
    ]
    config = PixelDiffConfig(
        local_similarity_iou_threshold=0.80,
        local_similarity_padding=2,
        local_similarity_search_radius=3,
        sparse_residual_max_area=0,
        small_residual_max_area=0,
    )

    filtered = filter_locally_similar_regions(regions, scan, template, config)

    assert [region.id for region in filtered] == [1]
    assert [(region.x, region.y) for region in filtered] == [(25, 9)]


def test_filter_locally_similar_regions_removes_horizontal_line_residuals() -> None:
    template = np.full((80, 160), 255, dtype=np.uint8)
    scan = np.full((80, 160), 255, dtype=np.uint8)
    cv2.line(template, (20, 40), (130, 40), 0, 2)
    cv2.line(scan, (20, 41), (130, 41), 0, 2)
    regions = [
        DifferenceRegion(id=1, x=20, y=38, width=112, height=8, area=800.0),
        DifferenceRegion(id=2, x=50, y=55, width=20, height=20, area=300.0),
    ]
    cv2.rectangle(scan, (50, 55), (68, 73), 0, -1)
    config = PixelDiffConfig(
        local_similarity_iou_threshold=1.0,
        horizontal_residual_min_aspect=10.0,
        horizontal_residual_max_height=12,
    )

    filtered = filter_locally_similar_regions(regions, scan, template, config)

    assert [(region.id, region.x, region.y) for region in filtered] == [(1, 50, 55)]


def test_filter_locally_similar_regions_removes_short_horizontal_residuals() -> None:
    template = np.full((80, 120), 255, dtype=np.uint8)
    scan = np.full((80, 120), 255, dtype=np.uint8)
    cv2.rectangle(template, (20, 30), (59, 39), 0, -1)
    cv2.rectangle(scan, (20, 30), (49, 39), 0, -1)
    regions = [
        DifferenceRegion(id=1, x=18, y=28, width=44, height=14, area=520.0),
        DifferenceRegion(id=2, x=70, y=28, width=18, height=22, area=320.0),
    ]
    cv2.rectangle(scan, (70, 28), (86, 48), 0, -1)
    config = PixelDiffConfig(
        local_similarity_iou_threshold=0.90,
        local_similarity_search_radius=0,
        short_horizontal_residual_min_aspect=2.5,
        short_horizontal_residual_max_height=20,
        short_horizontal_residual_min_iou=0.55,
    )

    filtered = filter_locally_similar_regions(regions, scan, template, config)

    assert [(region.id, region.x, region.y) for region in filtered] == [(1, 70, 28)]


def test_filter_locally_similar_regions_removes_wide_text_line_residuals() -> None:
    template = np.full((100, 260), 255, dtype=np.uint8)
    scan = np.full((100, 260), 255, dtype=np.uint8)
    cv2.putText(template, "LONG TEXT LINE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.putText(scan, "LONG TEXT  LINE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.rectangle(scan, (35, 70), (70, 90), 0, -1)
    regions = [
        DifferenceRegion(id=1, x=18, y=24, width=210, height=36, area=6200.0),
        DifferenceRegion(id=2, x=35, y=70, width=36, height=21, area=700.0),
    ]
    config = PixelDiffConfig(
        local_similarity_iou_threshold=0.90,
        wide_text_residual_min_area=5000.0,
        wide_text_residual_min_aspect=3.0,
        wide_text_residual_min_iou=0.30,
    )

    filtered = filter_locally_similar_regions(regions, scan, template, config)

    assert [(region.id, region.x, region.y) for region in filtered] == [(1, 35, 70)]


def test_filter_locally_similar_regions_removes_sparse_small_residuals() -> None:
    template = np.full((100, 100), 255, dtype=np.uint8)
    scan = np.full((100, 100), 255, dtype=np.uint8)
    cv2.rectangle(scan, (80, 80), (86, 86), 0, -1)
    cv2.putText(template, "A", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    cv2.putText(scan, "B", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    regions = [
        DifferenceRegion(id=1, x=78, y=78, width=11, height=11, area=220.0),
        DifferenceRegion(id=2, x=15, y=25, width=24, height=30, area=500.0),
    ]
    config = PixelDiffConfig(
        local_similarity_iou_threshold=1.0,
        sparse_residual_max_area=400.0,
        sparse_residual_max_density=0.04,
    )

    filtered = filter_locally_similar_regions(regions, scan, template, config)

    assert [(region.id, region.x, region.y) for region in filtered] == [(1, 15, 25)]
