from __future__ import annotations

import cv2
import numpy as np

from pixel_diff import morphology
from pixel_diff.models import PixelDiffConfig
from pixel_diff.morphology import clean_difference_mask, remove_small_components


def test_clean_difference_mask_can_group_nearby_difference_pixels() -> None:
    mask = np.zeros((80, 100), dtype=np.uint8)
    cv2.rectangle(mask, (20, 30), (25, 35), 255, -1)
    cv2.rectangle(mask, (32, 30), (37, 35), 255, -1)
    config = PixelDiffConfig(
        open_kernel=(1, 1),
        close_kernel=(1, 1),
        dilate_kernel=(15, 3),
        morph_iterations_open=0,
        morph_iterations_close=0,
        morph_iterations_dilate=1,
    )

    cleaned = clean_difference_mask(mask, config)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1


def test_remove_small_components_filters_isolated_scan_noise() -> None:
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5, 5] = 255
    mask[10:15, 10:15] = 255

    cleaned = remove_small_components(mask, min_area=5)

    assert cleaned[5, 5] == 0
    assert cleaned[12, 12] == 255


def test_clean_difference_mask_removes_tiny_noise_before_dilation() -> None:
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[5, 5] = 255
    cv2.rectangle(mask, (25, 25), (30, 30), 255, -1)
    config = PixelDiffConfig(
        min_noise_component_area=5,
        open_kernel=(1, 1),
        close_kernel=(1, 1),
        dilate_kernel=(9, 9),
        morph_iterations_open=0,
        morph_iterations_close=0,
        morph_iterations_dilate=1,
    )

    cleaned = clean_difference_mask(mask, config)

    assert cleaned[5, 5] == 0
    assert cleaned[27, 27] == 255


def test_component_keep_lookup_matches_area_threshold() -> None:
    stats = np.zeros((5, cv2.CC_STAT_MAX), dtype=np.int32)
    stats[:, cv2.CC_STAT_AREA] = [900, 1, 5, 6, 20]

    keep = morphology._component_keep_lookup(stats, min_area=6)

    assert keep.tolist() == [False, False, False, True, True]


def test_prepare_difference_masks_reuses_clean_result_without_text_mask(
    monkeypatch,
) -> None:
    diff = np.zeros((20, 20), dtype=np.uint8)
    diff[5:10, 5:10] = 255
    calls = 0
    real_clean = morphology.clean_difference_mask

    def counting_clean(mask: np.ndarray, config: PixelDiffConfig) -> np.ndarray:
        nonlocal calls
        calls += 1
        return real_clean(mask, config)

    monkeypatch.setattr(morphology, "clean_difference_mask", counting_clean)

    detection, recall = morphology.prepare_difference_masks(
        diff,
        unchanged_text_mask=None,
        config=PixelDiffConfig(),
    )

    assert calls == 1
    assert detection is recall
