from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.filter_pipeline import apply_multilevel_filters
from pixel_diff.models import DifferenceRegion, PixelDiffConfig


def test_multilevel_filter_uses_ssim_when_enabled() -> None:
    template = np.full((140, 360, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.putText(template, "SAME TEXT", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(scan, "SAME TEXT", (42, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.rectangle(scan, (270, 95), (310, 122), (0, 0, 0), -1)
    regions = [
        DifferenceRegion(id=1, x=35, y=45, width=180, height=45, area=5000.0),
        DifferenceRegion(id=2, x=270, y=95, width=41, height=28, area=1000.0),
    ]
    config = PixelDiffConfig(
        local_similarity_filter=False,
        multilevel_filter_enabled=True,
        ssim_filter_enabled=True,
        ssim_filter_threshold=0.90,
        ssim_filter_padding=12,
        ssim_filter_search_radius=3,
        ssim_filter_min_region_area=400,
    )

    result = apply_multilevel_filters(
        regions=regions,
        aligned_bgr=scan,
        template_bgr=template,
        scan_binary=_binary(scan),
        template_binary=_binary(template),
        scan_path="scan.png",
        template_path="template.png",
        page=0,
        config=config,
    )

    assert [region.id for region in result.regions] == [1]
    assert result.regions[0].x == 270
    assert result.metrics["filter_level2_removed"] == 1
    assert result.metrics["filter_ssim_checked"] == 2
    assert result.metrics["filter_ssim_max_score"] >= config.ssim_filter_threshold


def test_multilevel_filter_skips_ssim_when_pipeline_disabled() -> None:
    template = np.full((120, 280, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.putText(template, "ABC", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(scan, "ABC", (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    region = DifferenceRegion(id=1, x=25, y=40, width=90, height=45, area=1200.0)

    result = apply_multilevel_filters(
        regions=[region],
        aligned_bgr=scan,
        template_bgr=template,
        scan_binary=_binary(scan),
        template_binary=_binary(template),
        scan_path="scan.png",
        template_path="template.png",
        page=0,
        config=PixelDiffConfig(local_similarity_filter=False, multilevel_filter_enabled=False),
    )

    assert len(result.regions) == 1
    assert result.metrics["filter_level2_removed"] == 0
    assert result.metrics["filter_ssim_checked"] == 0


def test_level1_filters_isolated_small_residuals_when_multilevel_enabled() -> None:
    image = np.full((220, 420, 3), 255, dtype=np.uint8)
    regions = [
        DifferenceRegion(id=1, x=80, y=70, width=34, height=36, area=900.0),
        DifferenceRegion(id=2, x=260, y=140, width=80, height=45, area=2500.0),
    ]
    config = PixelDiffConfig(
        local_similarity_filter=False,
        multilevel_filter_enabled=True,
        ssim_filter_enabled=False,
        isolated_residual_filter_enabled=True,
        isolated_residual_max_area=1300,
        isolated_residual_max_width=45,
        isolated_residual_max_height=45,
        isolated_residual_min_neighbor_distance=90,
    )

    result = apply_multilevel_filters(
        regions=regions,
        aligned_bgr=image,
        template_bgr=image,
        scan_binary=_binary(image),
        template_binary=_binary(image),
        scan_path="scan.png",
        template_path="template.png",
        page=0,
        config=config,
    )

    assert [(region.x, region.y) for region in result.regions] == [(260, 140)]
    assert result.metrics["filter_level1_removed"] == 1


def test_level1_filters_colored_template_residuals_when_multilevel_enabled() -> None:
    template = np.full((180, 360, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.putText(template, "RED", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.putText(scan, "RED", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.rectangle(scan, (240, 105), (282, 135), (0, 0, 0), -1)
    regions = [
        DifferenceRegion(id=1, x=38, y=50, width=80, height=45, area=2000.0),
        DifferenceRegion(id=2, x=240, y=105, width=43, height=31, area=1300.0),
    ]
    config = PixelDiffConfig(
        local_similarity_filter=False,
        multilevel_filter_enabled=True,
        ssim_filter_enabled=False,
        colored_residual_filter_enabled=True,
        colored_residual_min_ratio=0.10,
        same_line_merge_filter_enabled=False,
    )

    result = apply_multilevel_filters(
        regions=regions,
        aligned_bgr=scan,
        template_bgr=template,
        scan_binary=_binary(scan),
        template_binary=_binary(template),
        scan_path="scan.png",
        template_path="template.png",
        page=0,
        config=config,
    )

    assert [(region.x, region.y) for region in result.regions] == [(240, 105)]
    assert result.metrics["filter_level1_removed"] == 1


def test_level1_merges_nearby_same_line_small_residuals_when_multilevel_enabled() -> None:
    image = np.full((220, 420, 3), 255, dtype=np.uint8)
    regions = [
        DifferenceRegion(id=1, x=100, y=90, width=20, height=18, area=240.0),
        DifferenceRegion(id=2, x=130, y=88, width=70, height=42, area=2100.0),
    ]
    config = PixelDiffConfig(
        local_similarity_filter=False,
        multilevel_filter_enabled=True,
        ssim_filter_enabled=False,
        same_line_merge_filter_enabled=True,
        same_line_merge_max_gap=36,
        same_line_merge_max_center_y_delta=18,
        same_line_merge_small_area=1300,
    )

    result = apply_multilevel_filters(
        regions=regions,
        aligned_bgr=image,
        template_bgr=image,
        scan_binary=_binary(image),
        template_binary=_binary(image),
        scan_path="scan.png",
        template_path="template.png",
        page=0,
        config=config,
    )

    assert len(result.regions) == 1
    assert result.regions[0].x == 100
    assert result.regions[0].width == 100
    assert result.metrics["filter_level1_removed"] == 1


def _binary(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
