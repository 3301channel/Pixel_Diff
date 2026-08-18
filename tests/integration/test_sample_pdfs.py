from __future__ import annotations

from pathlib import Path

import pytest

from pixel_diff import InputError, PixelDiffConfig
from pixel_diff.engine import PixelDiffEngine

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("scan_name", "template_name"),
    [
        ("document1_v3.pdf", "document1_v1.pdf"),
        ("document1_v4.pdf", "document1_v1.pdf"),
        ("document2_v2.pdf", "document2_v1.pdf"),
    ],
)
def test_sample_pdf_pairs_complete_or_fail_with_domain_error(
    scan_name: str,
    template_name: str,
    tmp_path: Path,
) -> None:
    scan_path = ROOT / scan_name
    template_path = ROOT / template_name
    if not scan_path.exists() or not template_path.exists():
        pytest.skip("local sample PDFs are not present")

    config = PixelDiffConfig(
        dpi=150,
        crop_margin=20,
        min_good_matches=10,
        min_diff_area=80,
        surf_hessian_threshold=400.0,
        dilate_kernel=(11, 8),
    )
    try:
        result = PixelDiffEngine(config).compare(
            scan_path,
            template_path,
            visual_output_path=tmp_path / f"{scan_name}_visual.png",
        )
    except InputError as exc:
        if "PyMuPDF is unavailable" in str(exc):
            pytest.skip(str(exc))
        raise

    assert result.status == "completed"
    assert result.image["width"] > 0
    assert result.image["height"] > 0
    assert isinstance(result.differences, list)


def test_document1_v2_filters_header_residuals_and_keeps_swift_change() -> None:
    scan_path = ROOT / "test_pdf" / "document1_v2.pdf"
    template_path = ROOT / "test_pdf" / "document1_v1.pdf"
    if not scan_path.exists() or not template_path.exists():
        pytest.skip("local sample PDFs are not present")

    result = PixelDiffEngine(PixelDiffConfig()).compare(scan_path, template_path)

    assert result.status == "completed"
    assert not any(region.y < 760 for region in result.differences)
    assert any(
        1080 <= region.x <= 1140 and 880 <= region.y <= 930
        for region in result.differences
    )
