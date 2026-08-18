from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.displacement import pair_displaced_regions
from pixel_diff.models import DifferenceRegion, PixelDiffConfig


def _region(region_id: int, x: int, y: int, width: int = 16, height: int = 16) -> DifferenceRegion:
    return DifferenceRegion(region_id, x, y, width, height, float(width * height))


def _page_pair(*, different_shape: bool = False) -> tuple[np.ndarray, np.ndarray]:
    scan = np.full((100, 140), 255, dtype=np.uint8)
    template = scan.copy()
    cv2.line(template, (23, 22), (34, 33), 0, 3)
    cv2.line(template, (34, 22), (23, 33), 0, 3)
    if different_shape:
        cv2.circle(scan, (94, 28), 6, 0, 2)
    else:
        cv2.line(scan, (88, 22), (99, 33), 0, 3)
        cv2.line(scan, (99, 22), (88, 33), 0, 3)
    return scan, template


def test_pairs_identical_added_and_deleted_shapes_as_one_displacement() -> None:
    scan, template = _page_pair()
    regions = [_region(1, 20, 19), _region(2, 85, 19, 18, 16)]

    paired, count = pair_displaced_regions(
        regions,
        scan,
        template,
        PixelDiffConfig(displacement_pairing_enabled=True),
    )

    assert count == 1
    assert len(paired) == 1
    assert paired[0].risk_reason == "content_displacement"
    assert paired[0].change_type == "displaced"
    assert paired[0].change_label == "文本错位"
    assert paired[0].classification_confidence is not None
    assert paired[0].classification_confidence >= 0.82
    assert (paired[0].x, paired[0].y, paired[0].width, paired[0].height) == (20, 19, 83, 16)


def test_does_not_pair_different_shapes() -> None:
    scan, template = _page_pair(different_shape=True)
    regions = [_region(1, 20, 19), _region(2, 85, 19, 18, 16)]

    paired, count = pair_displaced_regions(
        regions,
        scan,
        template,
        PixelDiffConfig(displacement_pairing_enabled=True),
    )

    assert count == 0
    assert len(paired) == 2


def test_does_not_pair_regions_with_the_same_direction() -> None:
    scan = np.full((100, 140), 255, dtype=np.uint8)
    template = scan.copy()
    cv2.rectangle(scan, (20, 20), (34, 34), 0, -1)
    cv2.rectangle(scan, (85, 20), (99, 34), 0, -1)
    regions = [_region(1, 19, 19, 17, 17), _region(2, 84, 19, 17, 17)]

    paired, count = pair_displaced_regions(
        regions,
        scan,
        template,
        PixelDiffConfig(displacement_pairing_enabled=True),
    )

    assert count == 0
    assert len(paired) == 2


def test_disabled_pairing_preserves_regions() -> None:
    scan, template = _page_pair()
    regions = [_region(7, 20, 19), _region(8, 85, 19, 18, 16)]

    paired, count = pair_displaced_regions(
        regions,
        scan,
        template,
        PixelDiffConfig(displacement_pairing_enabled=False),
    )

    assert count == 0
    assert paired == regions
