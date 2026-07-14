from __future__ import annotations

import pytest

from pixel_diff import ConfigurationError, PixelDiffConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dpi": 0},
        {"adaptive_block_size": 10},
        {"adaptive_block_size": 1},
        {"median_blur_kernel": 2},
        {"median_blur_kernel": 1},
        {"open_kernel": (0, 3)},
        {"crop_margin": -1},
        {"min_diff_area": -0.1},
        {"local_similarity_iou_threshold": -0.1},
        {"local_similarity_iou_threshold": 1.1},
        {"local_similarity_padding": -1},
        {"local_similarity_search_radius": -1},
        {"horizontal_residual_min_aspect": -0.1},
        {"horizontal_residual_max_height": -1},
        {"short_horizontal_residual_min_aspect": -0.1},
        {"short_horizontal_residual_max_height": -1},
        {"short_horizontal_residual_min_iou": -0.1},
        {"short_horizontal_residual_min_iou": 1.1},
        {"wide_text_residual_min_area": -0.1},
        {"wide_text_residual_min_aspect": -0.1},
        {"wide_text_residual_min_iou": -0.1},
        {"wide_text_residual_min_iou": 1.1},
        {"sparse_residual_max_area": -0.1},
        {"sparse_residual_max_density": -0.1},
        {"small_residual_max_area": -0.1},
        {"small_residual_max_density": -0.1},
        {"residual_filter_min_area": -0.1},
        {"residual_density_padding": -1},
        {"min_noise_component_area": -0.1},
        {"lowe_ratio": 1.0},
        {"lowe_ratio": 0.0},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    config = PixelDiffConfig(**kwargs)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_config_rejects_crop_margin_too_large_for_image() -> None:
    config = PixelDiffConfig(crop_margin=50)
    with pytest.raises(ConfigurationError):
        config.validate_for_image(width=100, height=200)


def test_default_config_is_valid() -> None:
    PixelDiffConfig().validate_for_image(width=2480, height=3508)
