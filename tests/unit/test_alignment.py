from __future__ import annotations

import numpy as np
import pytest

import pixel_diff.alignment as alignment_module
from pixel_diff import AlignmentError, PixelDiffConfig
from pixel_diff.alignment import (
    HomographyEstimate,
    _find_homography_deterministic,
    _restore_homography_scale,
    align_scan_to_template_bgr,
)


def test_alignment_rejects_blank_pages_without_descriptors() -> None:
    blank = np.full((120, 120, 3), 255, dtype=np.uint8)

    with pytest.raises(AlignmentError):
        align_scan_to_template_bgr(blank, blank, PixelDiffConfig(min_good_matches=4))


def test_alignment_ignores_pair_of_blank_pages_when_enabled() -> None:
    blank = np.full((120, 160, 3), 255, dtype=np.uint8)

    result = align_scan_to_template_bgr(
        blank,
        blank,
        PixelDiffConfig(blank_page_alignment_enabled=True),
    )

    assert result.blank_page_alignment
    assert result.blank_page_pair
    assert result.detector == "blank_identity"
    assert np.array_equal(result.aligned_bgr, blank)
    assert np.array_equal(result.homography, np.eye(3))


def test_alignment_uses_identity_when_only_template_is_blank() -> None:
    blank = np.full((120, 160, 3), 255, dtype=np.uint8)
    content = blank.copy()
    content[40:80, 60:100] = 0

    result = align_scan_to_template_bgr(
        content,
        blank,
        PixelDiffConfig(blank_page_alignment_enabled=True),
    )

    assert result.blank_page_alignment
    assert not result.blank_page_pair
    assert np.array_equal(result.aligned_bgr, content)


def test_restore_homography_from_reduced_feature_coordinates() -> None:
    reduced_homography = np.array(
        [[1.0, 0.02, 12.0], [-0.01, 1.0, -8.0], [0.0001, 0.0002, 1.0]],
        dtype=np.float64,
    )
    scale = 0.5
    scaling = np.diag([scale, scale, 1.0])

    restored = _restore_homography_scale(
        reduced_homography,
        scan_scale=scale,
        template_scale=scale,
    )

    expected = np.linalg.inv(scaling) @ reduced_homography @ scaling
    assert np.allclose(restored, expected)


def test_downsampled_alignment_restores_full_size_homography(monkeypatch) -> None:
    image = np.full((120, 200, 3), 255, dtype=np.uint8)
    seen_shapes: list[tuple[int, int]] = []

    def fake_estimate(
        scan_gray: np.ndarray,
        template_gray: np.ndarray,
        config: PixelDiffConfig,
        reprojection_threshold: float,
    ) -> HomographyEstimate:
        seen_shapes.append(scan_gray.shape)
        return HomographyEstimate(
            homography=np.array(
                [[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            good_matches=20,
            inlier_ratio=0.9,
            detector="sift",
            detector_fallback=False,
        )

    monkeypatch.setattr(alignment_module, "_estimate_homography", fake_estimate)
    result = align_scan_to_template_bgr(
        image,
        image,
        PixelDiffConfig(
            alignment_feature_downsample_enabled=True,
            alignment_feature_scale=0.5,
        ),
    )

    assert seen_shapes == [(60, 100)]
    assert result.feature_downsampled
    assert not result.feature_downsample_fallback
    assert result.feature_scale == 0.5
    assert np.allclose(result.homography[:2, 2], [10.0, 6.0])


def test_downsampled_alignment_falls_back_when_inlier_ratio_is_low(monkeypatch) -> None:
    image = np.full((120, 200, 3), 255, dtype=np.uint8)
    seen_shapes: list[tuple[int, int]] = []

    def fake_estimate(
        scan_gray: np.ndarray,
        template_gray: np.ndarray,
        config: PixelDiffConfig,
        reprojection_threshold: float,
    ) -> HomographyEstimate:
        seen_shapes.append(scan_gray.shape)
        return HomographyEstimate(
            homography=np.eye(3, dtype=np.float64),
            good_matches=20,
            inlier_ratio=0.2 if len(seen_shapes) == 1 else 0.9,
            detector="sift",
            detector_fallback=False,
        )

    monkeypatch.setattr(alignment_module, "_estimate_homography", fake_estimate)
    result = align_scan_to_template_bgr(
        image,
        image,
        PixelDiffConfig(
            alignment_feature_downsample_enabled=True,
            alignment_feature_scale=0.5,
            alignment_feature_fallback_enabled=True,
            alignment_feature_min_inlier_ratio=0.4,
        ),
    )

    assert seen_shapes == [(60, 100), (120, 200)]
    assert not result.feature_downsampled
    assert result.feature_downsample_fallback


def test_homography_estimation_is_independent_of_prior_opencv_rng_state() -> None:
    rng = np.random.default_rng(20260720)
    scan_points = rng.uniform(0, 1000, size=(80, 1, 2)).astype(np.float32)
    template_points = scan_points.copy()
    template_points[:, 0, 0] += 12.0
    template_points[:, 0, 1] -= 7.0
    template_points[:25] = rng.uniform(0, 1000, size=(25, 1, 2))

    alignment_module.cv2.setRNGSeed(1)
    first_h, first_mask = _find_homography_deterministic(
        scan_points,
        template_points,
        reprojection_threshold=3.0,
    )
    alignment_module.cv2.setRNGSeed(987654)
    second_h, second_mask = _find_homography_deterministic(
        scan_points,
        template_points,
        reprojection_threshold=3.0,
    )

    assert first_h is not None and second_h is not None
    assert first_mask is not None and second_mask is not None
    assert np.allclose(first_h, second_h)
    assert np.array_equal(first_mask, second_mask)
