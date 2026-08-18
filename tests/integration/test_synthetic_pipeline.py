from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pixel_diff.engine import PixelDiffEngine
from pixel_diff.models import PixelDiffConfig


def test_engine_detects_geometric_text_like_change_on_synthetic_images(tmp_path: Path) -> None:
    template = _synthetic_page()
    changed = template.copy()
    cv2.putText(changed, ".", (450, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    scan = _warp_like_scan(changed)
    template_path = tmp_path / "template.png"
    scan_path = tmp_path / "scan.png"
    visual_path = tmp_path / "visual.png"
    template_output_path = tmp_path / "page_0001_original.png"
    candidate_output_path = tmp_path / "page_0001_candidate.png"
    ghost_output_path = tmp_path / "page_0001_heatmap.png"
    cv2.imwrite(str(template_path), template)
    cv2.imwrite(str(scan_path), scan)

    config = PixelDiffConfig(
        dpi=150,
        min_good_matches=8,
        crop_margin=5,
        min_diff_area=20,
        surf_hessian_threshold=400.0,
        dilate_kernel=(9, 6),
    )
    result = PixelDiffEngine(config).compare(
        scan_path,
        template_path,
        visual_output_path=visual_path,
        template_output_path=template_output_path,
        candidate_output_path=candidate_output_path,
        ghost_output_path=ghost_output_path,
    )

    assert result.status == "completed"
    assert result.image == {"width": 600, "height": 800, "dpi": 150}
    assert result.differences
    assert visual_path.exists()
    assert template_output_path.exists()
    assert candidate_output_path.exists()
    assert ghost_output_path.exists()
    assert result.metadata["template_output_path"] == str(template_output_path)
    assert result.metadata["candidate_output_path"] == str(candidate_output_path)
    assert result.metadata["ghost_output_path"] == str(ghost_output_path)
    assert result.metrics["good_matches"] >= config.min_good_matches
    assert result.metrics["feature_detector"] in {"surf", "sift"}


def _synthetic_page() -> np.ndarray:
    image = np.full((800, 600, 3), 255, dtype=np.uint8)
    for index in range(18):
        y = 70 + index * 35
        cv2.putText(
            image,
            f"Pixel Diff Contract Line {index:02d}",
            (50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(image, (500, y - 8), 5 + index % 4, (0, 0, 0), -1)
    cv2.rectangle(image, (45, 45), (555, 755), (0, 0, 0), 2)
    return image


def _warp_like_scan(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 1.5, 1.01)
    matrix[:, 2] += (8, -5)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
