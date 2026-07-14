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
