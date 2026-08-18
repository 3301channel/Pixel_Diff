from __future__ import annotations

from pathlib import Path

import pytest

from pixel_diff import ConfigurationError
from pixel_diff.models import DifferenceRegion, PixelDiffConfig, PixelDiffResult


def test_config_from_yaml_loads_kernel_lists(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dpi: 200\nopen_kernel: [1, 2]\nclose_kernel: [3, 4]\ndilate_kernel: [5, 6]\n",
        encoding="utf-8",
    )

    config = PixelDiffConfig.from_yaml(config_path)

    assert config.dpi == 200
    assert config.open_kernel == (1, 2)
    assert config.close_kernel == (3, 4)
    assert config.dilate_kernel == (5, 6)


def test_config_rejects_invalid_match_and_ransac_values() -> None:
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(feature_detector="orb").validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(feature_detector_fallback="orb").validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(surf_hessian_threshold=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(sift_nfeatures=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_centroid_max_drift=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_centroid_row_dilate_width=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_centroid_row_dilate_height=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_centroid_min_width_ratio=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_centroid_max_height=8, line_centroid_min_height=8).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_centroid_median_tolerance=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_horizontal_max_shift=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_horizontal_band_half_height=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_horizontal_min_iou=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_horizontal_min_improvement=-0.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_affine_window_width=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_affine_min_anchors=1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_affine_max_scale_delta=0.0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(line_affine_min_improvement=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(residual_line_window_width=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(residual_line_min_anchor_similarity=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(residual_line_min_diff_reduction=-0.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(residual_line_min_protected_retention=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_region_padding=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_ocr_padding=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_ocr_match_ratio=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_plain_text_max_area=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_plain_text_ssim_threshold=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_ssim_padding=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_ssim_search_radius=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(patch_export_padding=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(displacement_pairing_min_direction_ratio=0.49).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(displacement_pairing_min_similarity=1.01).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(displacement_pairing_max_size_ratio=0.99).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(displacement_pairing_padding=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_large_visual_min_area=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_large_visual_min_width=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_large_visual_min_height=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_watermark_max_p95_delta=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_watermark_max_very_dark_ratio=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_watermark_max_template_dark_ratio=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_watermark_dark_threshold=256).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_watermark_template_dark_threshold=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(local_warp_max_displacement=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(local_warp_scale=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(local_warp_scale=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(local_warp_blur_kernel=2).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(local_warp_gate_min_iou=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(alignment_feature_scale=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(alignment_feature_scale=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(alignment_feature_min_inlier_ratio=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(report_parallel_workers=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(rigid_text_block_min_gap_width=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(rigid_text_block_max_internal_gap=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(rigid_text_block_min_anchor_similarity=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(rigid_text_block_min_iou_improvement=-0.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(ghost_match_tolerance=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_stroke_match_tolerance=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_stroke_match_min_coverage=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(
            risk_review_stroke_match_min_area=10,
            risk_review_stroke_match_max_area=9,
        ).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_narrow_stroke_max_area=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_narrow_stroke_max_width=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_narrow_stroke_max_height=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_narrow_stroke_min_coverage=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_page_number_bottom_ratio=1.0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(risk_review_page_number_shape_tolerance=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(ssim_filter_threshold=1.1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(ssim_filter_padding=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(ssim_filter_search_radius=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(ssim_filter_min_region_area=-1).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(min_good_matches=3).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(ransac_reprojection_threshold=0).validate()
    with pytest.raises(ConfigurationError):
        PixelDiffConfig(morph_iterations_open=-1).validate()


def test_result_to_dict_is_json_ready() -> None:
    result = PixelDiffResult(
        status="completed",
        page=0,
        image={"width": 10, "height": 20, "dpi": 300},
        differences=[DifferenceRegion(1, 2, 3, 4, 5, 6.0)],
        metrics={"elapsed_ms": 1},
        visual_output_path=None,
        metadata={"sample": "x"},
    )

    assert result.to_dict() == {
        "status": "completed",
        "page": 0,
        "image": {"width": 10, "height": 20, "dpi": 300},
        "differences": [{"id": 1, "x": 2, "y": 3, "width": 4, "height": 5, "area": 6.0}],
        "metrics": {"elapsed_ms": 1},
        "visual_output_path": None,
        "metadata": {"sample": "x"},
    }
