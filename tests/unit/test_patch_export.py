from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pixel_diff.models import DifferenceRegion, PixelDiffConfig
from pixel_diff.patch_export import export_region_patches


def test_export_region_patches_writes_template_scan_diff_and_metadata(tmp_path: Path) -> None:
    template = np.full((80, 120, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.rectangle(template, (20, 20), (45, 45), (0, 0, 0), -1)
    cv2.rectangle(scan, (22, 20), (47, 45), (0, 0, 0), -1)
    region = DifferenceRegion(
        id=1,
        x=18,
        y=18,
        width=34,
        height=32,
        area=800.0,
        risk_level="MEDIUM",
        risk_reason="template_text_overlap",
        template_text="普通文本",
    )

    export_dir = export_region_patches(
        regions=[region],
        template_bgr=template,
        scan_bgr=scan,
        output_dir=tmp_path / "patches",
        page=0,
        config=PixelDiffConfig(patch_export_enabled=True, patch_export_padding=6),
    )

    assert export_dir == tmp_path / "patches"
    assert (export_dir / "page_0001_region_0001_template.png").exists()
    assert (export_dir / "page_0001_region_0001_scan.png").exists()
    assert (export_dir / "page_0001_region_0001_diff.png").exists()
    metadata = json.loads((export_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["page"] == 1
    assert metadata["regions"][0]["id"] == 1
    assert metadata["regions"][0]["risk_level"] == "MEDIUM"
    assert metadata["regions"][0]["template_text"] == "普通文本"


def test_export_region_patches_is_noop_when_disabled(tmp_path: Path) -> None:
    image = np.full((80, 120, 3), 255, dtype=np.uint8)

    export_dir = export_region_patches(
        regions=[DifferenceRegion(id=1, x=18, y=18, width=34, height=32, area=800.0)],
        template_bgr=image,
        scan_bgr=image,
        output_dir=tmp_path / "patches",
        page=0,
        config=PixelDiffConfig(patch_export_enabled=False),
    )

    assert export_dir is None
    assert not (tmp_path / "patches").exists()
