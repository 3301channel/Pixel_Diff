from __future__ import annotations

import numpy as np

from pixel_diff.change_classification import classify_difference_regions
from pixel_diff.models import DifferenceRegion, PixelDiffConfig


def _region(region_id: int, x: int) -> DifferenceRegion:
    return DifferenceRegion(region_id, x, 0, 4, 4, 16.0)


def test_classifies_added_deleted_and_modified_regions() -> None:
    template = np.full((4, 12), 255, dtype=np.uint8)
    scan = template.copy()
    scan[:, 0:4] = 0
    template[:, 4:8] = 0
    scan[:, 8:10] = 0
    template[:, 10:12] = 0

    result = classify_difference_regions(
        [_region(1, 0), _region(2, 4), _region(3, 8)],
        scan,
        template,
        PixelDiffConfig(),
    )

    assert [item.change_type for item in result] == ["added", "deleted", "modified"]
    assert [item.change_label for item in result] == ["新增", "删除", "修改"]
    assert result[0].added_pixels == 16
    assert result[1].deleted_pixels == 16
    assert result[2].classification_confidence == 1.0


def test_zero_direction_pixels_fall_back_to_modified() -> None:
    blank = np.full((4, 4), 255, dtype=np.uint8)
    result = classify_difference_regions(
        [_region(1, 0)], blank, blank, PixelDiffConfig()
    )[0]

    assert result.change_type == "modified"
    assert result.change_label == "修改"
    assert result.classification_confidence == 0.0


def _balanced_page(
    region: DifferenceRegion, *, added_columns: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    scan = np.full((100, 200), 255, dtype=np.uint8)
    template = scan.copy()
    split = added_columns if added_columns is not None else region.width // 2
    scan[region.y : region.y + region.height, region.x : region.x + split] = 0
    template[
        region.y : region.y + region.height,
        region.x + split : region.x + region.width,
    ] = 0
    return scan, template


def _large_rule_config(**overrides: object) -> PixelDiffConfig:
    values = {
        "large_modified_as_displaced_enabled": True,
        "large_modified_min_page_area_ratio": 0.02,
        "large_modified_min_aspect_ratio": 6.0,
        "large_modified_min_direction_balance": 0.60,
    }
    values.update(overrides)
    return PixelDiffConfig(**values)


def test_large_wide_balanced_modified_region_becomes_displaced() -> None:
    region = DifferenceRegion(1, 20, 20, 120, 10, 1200.0)
    scan, template = _balanced_page(region)

    result = classify_difference_regions(
        [region], scan, template, _large_rule_config()
    )[0]

    assert result.change_type == "displaced"
    assert result.change_label == "文本错位"
    assert result.classification_reason == "large_balanced_horizontal_residual"


def test_large_displacement_upgrade_requires_every_gate() -> None:
    cases = [
        DifferenceRegion(1, 20, 20, 20, 10, 200.0),
        DifferenceRegion(2, 20, 20, 40, 20, 800.0),
    ]
    for region in cases:
        scan, template = _balanced_page(region)
        result = classify_difference_regions(
            [region], scan, template, _large_rule_config()
        )[0]
        assert result.change_type == "modified"

    region = DifferenceRegion(3, 20, 20, 120, 10, 1200.0)
    scan, template = _balanced_page(region, added_columns=84)
    result = classify_difference_regions(
        [region], scan, template, _large_rule_config()
    )[0]
    assert result.change_type == "modified"


def test_large_displacement_upgrade_can_be_disabled() -> None:
    region = DifferenceRegion(1, 20, 20, 120, 10, 1200.0)
    scan, template = _balanced_page(region)
    result = classify_difference_regions(
        [region],
        scan,
        template,
        _large_rule_config(large_modified_as_displaced_enabled=False),
    )[0]
    assert result.change_type == "modified"
