from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pixel_diff import InputError

ROOT = Path(__file__).resolve().parents[2]


def test_prepare_output_paths_places_files_in_per_run_directory(tmp_path: Path) -> None:
    compare_cli = _load_compare_cli()
    visual_path = tmp_path / "artifacts" / "result.png"
    json_path = tmp_path / "artifacts" / "result.json"
    ghost_path = tmp_path / "artifacts" / "heatmap.png"

    output_paths = compare_cli._prepare_output_paths(
        scan_path="document1_v3.pdf",
        template_path="document1_v1.pdf",
        visual_path=str(visual_path),
        json_path=str(json_path),
        ghost_path=str(ghost_path),
        report_dir=None,
        run_id="20260710_153000_000000",
    )

    assert output_paths.run_dir == (
        tmp_path / "artifacts" / "20260710_153000_000000_document1_v1_vs_document1_v3"
    )
    assert output_paths.run_dir.exists()
    assert output_paths.visual == output_paths.run_dir / "result.png"
    assert output_paths.json == output_paths.run_dir / "result.json"
    assert output_paths.ghost == output_paths.run_dir / "heatmap.png"


def test_prepare_output_paths_uses_unique_directory_for_duplicate_run_id(
    tmp_path: Path,
) -> None:
    compare_cli = _load_compare_cli()
    existing_dir = tmp_path / "20260710_153000_000000_template_vs_scan"
    existing_dir.mkdir()

    output_paths = compare_cli._prepare_output_paths(
        scan_path="scan.pdf",
        template_path="template.pdf",
        visual_path=str(tmp_path / "result.png"),
        json_path=None,
        ghost_path=None,
        report_dir=None,
        run_id="20260710_153000_000000",
    )

    assert output_paths.run_dir == tmp_path / "20260710_153000_000000_template_vs_scan_2"
    assert output_paths.visual == output_paths.run_dir / "result.png"
    assert output_paths.json == output_paths.run_dir / "diff_result.json"
    assert output_paths.ghost is None


def test_prepare_output_paths_writes_default_json_for_visual_outputs(tmp_path: Path) -> None:
    compare_cli = _load_compare_cli()

    output_paths = compare_cli._prepare_output_paths(
        scan_path="scan.pdf",
        template_path="template.pdf",
        visual_path=str(tmp_path / "result.png"),
        json_path=None,
        ghost_path=None,
        report_dir=None,
        run_id="20260710_153000_000000",
    )

    assert output_paths.json == output_paths.run_dir / "diff_result.json"


def test_single_result_summary_is_not_raw_json(tmp_path: Path) -> None:
    compare_cli = _load_compare_cli()
    output_paths = compare_cli.CliOutputPaths(
        visual=tmp_path / "result.png",
        json=tmp_path / "diff_result.json",
        ghost=tmp_path / "heatmap.png",
        template=None,
        candidate=None,
        report_dir=None,
        report_html=None,
        report_docx=None,
        run_dir=tmp_path,
    )
    result = _FakeResult(page=0, differences=[object(), object()])

    summary = compare_cli._format_single_result_summary(result, output_paths)

    assert "completed page=1 differences=2" in summary
    assert "json:" in summary
    assert "{" not in summary
    assert "differences\": [" not in summary


def test_prepare_output_paths_creates_report_layout(tmp_path: Path) -> None:
    compare_cli = _load_compare_cli()

    output_paths = compare_cli._prepare_output_paths(
        scan_path="document1_v3.pdf",
        template_path="document1_v1.pdf",
        visual_path=None,
        json_path=None,
        ghost_path=None,
        report_dir=str(tmp_path),
        run_id="20260710_153000_000000",
    )

    expected_run_dir = tmp_path / "20260710_153000_000000_document1_v1_vs_document1_v3"
    expected_report_dir = expected_run_dir / "report"
    assert output_paths.run_dir == expected_run_dir
    assert output_paths.json == expected_run_dir / "diff_result.json"
    assert output_paths.report_dir == expected_report_dir
    assert output_paths.report_html == expected_report_dir / "diff_report.html"
    assert output_paths.report_docx == expected_report_dir / "diff_report.docx"
    assert output_paths.template == expected_report_dir / "page_0001_original.png"
    assert output_paths.candidate == expected_report_dir / "page_0001_candidate.png"
    assert output_paths.ghost == expected_report_dir / "page_0001_heatmap.png"


def test_report_summary_includes_docx_path(tmp_path: Path) -> None:
    compare_cli = _load_compare_cli()
    output_paths = compare_cli.CliOutputPaths(
        visual=None,
        json=tmp_path / "diff_result.json",
        ghost=None,
        template=None,
        candidate=None,
        report_dir=tmp_path,
        report_html=tmp_path / "diff_report.html",
        report_docx=tmp_path / "diff_report.docx",
        run_dir=tmp_path,
    )
    payload = {"pages": [{"page": 1}], "total_regions": 3}

    summary = compare_cli._format_report_summary(payload, output_paths)

    assert "completed pages=1 differences=3" in summary
    assert "docx:" in summary
    assert "diff_report.docx" in summary


def test_report_page_output_paths_use_one_based_page_names(tmp_path: Path) -> None:
    compare_cli = _load_compare_cli()
    output_paths = compare_cli._prepare_output_paths(
        scan_path="document1_v3.pdf",
        template_path="document1_v1.pdf",
        visual_path=None,
        json_path=None,
        ghost_path=None,
        report_dir=str(tmp_path),
        run_id="20260710_153000_000000",
    )

    page_paths = compare_cli._report_page_output_paths(output_paths, page=1)

    assert page_paths["template_output_path"].endswith("page_0002_original.png")
    assert page_paths["candidate_output_path"].endswith("page_0002_candidate.png")
    assert page_paths["ghost_output_path"].endswith("page_0002_heatmap.png")


def test_resolve_pages_returns_all_pages_or_requested_page(tmp_path: Path) -> None:
    compare_cli = _load_compare_cli()
    scan_path = _write_pdf(tmp_path / "scan.pdf", pages=2)
    template_path = _write_pdf(tmp_path / "template.pdf", pages=2)

    assert compare_cli._resolve_pages(str(scan_path), str(template_path), None) == [0, 1]
    assert compare_cli._resolve_pages(str(scan_path), str(template_path), 1) == [1]


def test_resolve_pages_rejects_mismatched_page_counts(tmp_path: Path) -> None:
    compare_cli = _load_compare_cli()
    scan_path = _write_pdf(tmp_path / "scan.pdf", pages=2)
    template_path = _write_pdf(tmp_path / "template.pdf", pages=1)

    with pytest.raises(InputError):
        compare_cli._resolve_pages(str(scan_path), str(template_path), None)


def _load_compare_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_cli", ROOT / "scripts" / "compare.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_cli"] = module
    spec.loader.exec_module(module)
    return module


class _FakeResult:
    def __init__(self, page: int, differences: list[object]) -> None:
        self.page = page
        self.differences = differences


def _write_pdf(path: Path, pages: int) -> Path:
    fitz = _fitz_or_skip()
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=72, height=72)
        page.insert_text((10, 30), f"Page {index + 1}")
    document.save(path)
    document.close()
    return path


def _fitz_or_skip() -> object:
    try:
        import fitz
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PyMuPDF unavailable: {exc}")
    return fitz
