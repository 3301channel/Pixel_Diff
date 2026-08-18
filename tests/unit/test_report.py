from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from pixel_diff.models import DifferenceRegion, PixelDiffConfig, PixelDiffResult
from pixel_diff.report import (
    build_document_report_payload,
    render_html_report,
    write_docx_report,
)


def test_render_html_report_uses_chinese_visible_text(tmp_path: Path) -> None:
    payload = {
        "run_id": "20260710_153000_000000_document1_v1_vs_document1_v3",
        "template_path": "document1_v1.pdf",
        "scan_path": "document1_v3.pdf",
        "dpi": 300,
        "total_regions": 0,
        "difference_rate": 0.0,
        "metrics": {"good_matches": 12, "inlier_ratio": 0.75},
        "pages": [{"page": 1, "regions": 0, "outputs": {}}],
        "regions": [],
        "outputs": {"docx_output_path": str(tmp_path / "diff_report.docx")},
    }

    html = render_html_report(payload, report_dir=tmp_path)

    assert '<html lang="zh-CN">' in html
    assert "Pixel-Diff 像素级差异检测报告" in html
    assert "导出比对报告" in html
    assert "diff_report.docx" in html
    assert "文档差异率" in html
    assert "模板/审批通过文件" in html
    assert "待检/扫描文件" in html
    assert "本页未发现疑似差异区域。" in html


def test_build_document_report_payload_and_html_include_multiple_pages(
    tmp_path: Path,
) -> None:
    results = [
        PixelDiffResult(
            status="completed",
            page=0,
            image={"width": 10, "height": 20, "dpi": 300},
            differences=[DifferenceRegion(id=1, x=1, y=2, width=3, height=4, area=12.0)],
            metrics={
                "elapsed_ms": 10,
                "good_matches": 20,
                "inlier_ratio": 0.8,
                "line_horizontal_applied": 1,
                "line_horizontal_anchors": 4,
                "line_horizontal_max_abs_offset": 6.0,
                "alignment_feature_downsampled": 1,
                "alignment_feature_scale": 0.5,
                "alignment_feature_fallback": 0,
                "displacement_pairs": 1,
                "risk_review_narrow_stroke_filtered": 4,
                "rigid_text_blocks_attempted": 5,
                "rigid_text_blocks_applied": 3,
                "rigid_text_blocks_rejected_overlap": 1,
                "rigid_text_blocks_rejected_quality": 1,
                "rigid_text_block_applied_lines": 2,
            },
        ),
        PixelDiffResult(
            status="completed",
            page=1,
            image={"width": 10, "height": 20, "dpi": 300},
            differences=[DifferenceRegion(id=1, x=5, y=6, width=7, height=8, area=56.0)],
            metrics={"elapsed_ms": 15, "good_matches": 30, "inlier_ratio": 0.9},
        ),
    ]

    payload = build_document_report_payload(
        results,
        run_id="run",
        scan_path="scan.pdf",
        template_path="template.pdf",
        config=PixelDiffConfig(),
        output_paths_by_page={
            0: {"ghost_output_path": str(tmp_path / "page_0001_heatmap.png")},
            1: {"ghost_output_path": str(tmp_path / "page_0002_heatmap.png")},
        },
    )
    html = render_html_report(payload, report_dir=tmp_path)

    assert payload["total_pages"] == 2
    assert payload["total_regions"] == 2
    assert payload["total_diff_area"] == 68.0
    assert payload["total_page_area"] == 400.0
    assert payload["difference_rate"] == 0.17
    assert [page["page"] for page in payload["pages"]] == [1, 2]
    assert payload["pages"][0]["difference_rate"] == 0.06
    assert payload["pages"][0]["line_horizontal_applied"] == 1
    assert payload["pages"][0]["line_horizontal_anchors"] == 4
    assert payload["pages"][0]["line_horizontal_max_abs_offset"] == 6.0
    assert payload["pages"][0]["alignment_feature_downsampled"] == 1
    assert payload["pages"][0]["alignment_feature_scale"] == 0.5
    assert payload["pages"][0]["alignment_feature_fallback"] == 0
    assert payload["pages"][0]["displacement_pairs"] == 1
    assert payload["pages"][0]["risk_review_narrow_stroke_filtered"] == 4
    assert payload["pages"][0]["rigid_text_blocks_attempted"] == 5
    assert payload["pages"][0]["rigid_text_blocks_applied"] == 3
    assert payload["pages"][0]["rigid_text_block_applied_lines"] == 2
    assert payload["pages"][1]["difference_rate"] == 0.28
    assert "第 1 页" in html
    assert "第 2 页" in html
    assert "page_0002_heatmap.png" in html
    assert html.index("(1, 2)") < html.index("第 2 页")
    assert html.index("第 2 页") < html.index("(5, 6)")
    assert "本页差异坐标" in html


def test_write_docx_report_creates_downloadable_word_file(tmp_path: Path) -> None:
    payload = {
        "run_id": "run",
        "template_path": "template.pdf",
        "scan_path": "scan.pdf",
        "dpi": 300,
        "total_pages": 1,
        "total_regions": 1,
        "difference_rate": 0.0123,
        "pages": [
            {
                "page": 1,
                "regions": 1,
                "difference_rate": 0.0123,
                "outputs": {},
            }
        ],
        "regions": [
            {
                "page": 1,
                "id": 1,
                "x": 10,
                "y": 20,
                "width": 30,
                "height": 40,
                "area": 500.0,
            }
        ],
        "outputs": {},
    }
    output_path = tmp_path / "diff_report.docx"

    write_docx_report(payload, output_path)

    assert output_path.exists()
    with ZipFile(output_path) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "Pixel-Diff 像素级差异检测报告" in document_xml
    assert "文档差异率" in document_xml
    assert "第 1 页" in document_xml
