"""CLI 入口脚本 —— 命令行比对工具。

用法:
    python -m scripts.compare <scan_path> <template_path> [选项]

选项:
    --page N        指定比对页面（0-based），默认比对所有页
    --config PATH   配置文件路径（YAML），默认 configs/default.yaml
    --visual PATH   红框标注图输出路径
    --json PATH     JSON 结果输出路径
    --ghost PATH    文字残影图输出路径
    --report-dir DIR 报告模式：在此目录下生成完整报告包（HTML + DOCX + 图片）

报告模式（--report-dir）与单页输出模式互斥。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pixel_diff import InputError, PixelDiffConfig, PixelDiffError, compare
from pixel_diff.io import convert_docx_to_pdf, get_document_page_count, write_json
from pixel_diff.report import build_document_report_payload, render_html_report, write_docx_report


@dataclass(frozen=True)
class CliOutputPaths:
    """单次 CLI 比对运行的输出路径集合。"""

    visual: Path | None
    json: Path | None
    ghost: Path | None
    template: Path | None
    candidate: Path | None
    report_dir: Path | None
    report_html: Path | None
    report_docx: Path | None
    run_dir: Path | None


def main() -> int:
    """CLI 主函数。返回 0=成功，2=PixelDiff 域错误。"""
    parser = argparse.ArgumentParser(
        description="Compare a scan PDF/image with a template PDF/image."
    )
    parser.add_argument("scan_path")        # 待检文件路径
    parser.add_argument("template_path")    # 模板文件路径
    parser.add_argument("--page", type=int) # 指定页码
    parser.add_argument("--config", default="configs/default.yaml")  # 配置文件
    parser.add_argument("--visual")         # 红框标注图输出路径
    parser.add_argument("--json")           # JSON 结果输出路径
    parser.add_argument("--ghost", "--heatmap", dest="ghost")  # 残影图（别名 --heatmap）
    parser.add_argument(
        "--report-dir",
        help="Create a report-style output package under this directory.",
    )
    args = parser.parse_args()

    # 加载配置（YAML 不存在时回退默认）
    config = (
        PixelDiffConfig.from_yaml(args.config)
        if Path(args.config).exists()
        else PixelDiffConfig()
    )

    # 准备输出路径
    output_paths = _prepare_output_paths(
        scan_path=args.scan_path,
        template_path=args.template_path,
        visual_path=args.visual,
        json_path=args.json,
        ghost_path=args.ghost,
        report_dir=args.report_dir,
    )

    # DOCX 输入自动转 PDF
    scan_path, template_path = _prepare_comparable_inputs(
        args.scan_path,
        args.template_path,
        output_paths.run_dir,
    )

    # ── 报告模式：逐页比对 + 生成 HTML/DOCX 报告 ──
    if output_paths.report_dir is not None and output_paths.json is not None:
        pages = _resolve_pages(scan_path, template_path, args.page)
        results = []
        output_paths_by_page: dict[int, dict[str, str]] = {}
        for page in pages:
            page_paths = _report_page_output_paths(output_paths, page)
            result = compare(
                scan_path=scan_path,
                template_path=template_path,
                page=page,
                config=config,
                ghost_output_path=page_paths["ghost_output_path"],
                template_output_path=page_paths["template_output_path"],
                candidate_output_path=page_paths["candidate_output_path"],
            )
            results.append(result)
            output_paths_by_page[page] = {
                **page_paths,
                "json_output_path": str(output_paths.json),
            }
            if output_paths.report_html is not None:
                output_paths_by_page[page]["html_output_path"] = str(output_paths.report_html)
            if output_paths.report_docx is not None:
                output_paths_by_page[page]["docx_output_path"] = str(output_paths.report_docx)

        # 构建报告载荷
        report_payload = build_document_report_payload(
            results,
            run_id=output_paths.run_dir.name if output_paths.run_dir else _new_run_id(),
            scan_path=args.scan_path,
            template_path=args.template_path,
            config=config,
            output_paths_by_page=output_paths_by_page,
        )

        # 输出 JSON + HTML + DOCX
        write_json(output_paths.json, report_payload)
        if output_paths.report_html is not None:
            output_paths.report_html.parent.mkdir(parents=True, exist_ok=True)
            output_paths.report_html.write_text(
                render_html_report(report_payload, report_dir=output_paths.report_dir),
                encoding="utf-8",
            )
        if output_paths.report_docx is not None:
            write_docx_report(report_payload, output_paths.report_docx)

        print(_format_report_summary(report_payload, output_paths))
        return 0

    # ── 单页模式：快速比对 ──
    page = args.page if args.page is not None else 0
    result = compare(
        scan_path=scan_path,
        template_path=template_path,
        page=page,
        config=config,
        visual_output_path=output_paths.visual,
        json_output_path=output_paths.json,
        ghost_output_path=output_paths.ghost,
        template_output_path=output_paths.template,
        candidate_output_path=output_paths.candidate,
    )
    print(_format_single_result_summary(result, output_paths))
    return 0


def _prepare_output_paths(
    scan_path: str,
    template_path: str,
    visual_path: str | None,
    json_path: str | None,
    ghost_path: str | None,
    report_dir: str | None,
    run_id: str | None = None,
) -> CliOutputPaths:
    """根据用户指定的路径或报告目录，生成 CliOutputPaths 结构。

    输出文件放在以运行 ID + 文件名命名的独立子目录下，
    避免多次运行互相覆盖。
    """
    requested_paths = [
        Path(path)
        for path in (visual_path, json_path, ghost_path)
        if path is not None
    ]
    if not requested_paths and report_dir is None:
        # 未指定任何输出路径 → 仅返回结果，不写文件
        return CliOutputPaths(
            visual=None, json=None, ghost=None,
            template=None, candidate=None,
            report_dir=None, report_html=None, report_docx=None,
            run_dir=None,
        )

    base_dir = Path(report_dir) if report_dir is not None else requested_paths[0].parent
    run_name = "_".join(
        (
            run_id or _new_run_id(),
            _safe_path_stem(template_path),
            "vs",
            _safe_path_stem(scan_path),
        )
    )
    run_dir = _create_unique_run_dir(base_dir, run_name)
    page_report_dir = run_dir / "report" if report_dir is not None else None

    return CliOutputPaths(
        visual=run_dir / Path(visual_path).name if visual_path is not None else None,
        json=(
            run_dir / Path(json_path).name
            if json_path is not None
            else run_dir / "diff_result.json"
        ),
        ghost=(
            page_report_dir / "page_0001_heatmap.png"
            if page_report_dir is not None
            else run_dir / Path(ghost_path).name
            if ghost_path is not None
            else None
        ),
        template=(
            page_report_dir / "page_0001_original.png"
            if page_report_dir is not None
            else None
        ),
        candidate=(
            page_report_dir / "page_0001_candidate.png"
            if page_report_dir is not None
            else None
        ),
        report_dir=page_report_dir,
        report_html=page_report_dir / "diff_report.html" if page_report_dir is not None else None,
        report_docx=page_report_dir / "diff_report.docx" if page_report_dir is not None else None,
        run_dir=run_dir,
    )


def _new_run_id() -> str:
    """生成运行 ID：yyyyMMdd_HHmmss_ffffff（微秒级时间戳）。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _safe_path_stem(path: str) -> str:
    """将文件名转为安全的目录名（只保留字母数字和 -_ ）。"""
    stem = Path(path).stem or "output"
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem)
    return safe.strip("_") or "output"


def _create_unique_run_dir(base_dir: Path, run_name: str) -> Path:
    """创建唯一运行目录：若目录已存在，追加 _2、_3... 后缀。"""
    candidate = base_dir / run_name
    index = 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base_dir / f"{run_name}_{index}"
            index += 1


def _resolve_pages(scan_path: str, template_path: str, requested_page: int | None) -> list[int]:
    """解析需要比对的页面索引列表。

    单页模式：返回 [page]
    全文档模式：返回 range(0, pages)，要求扫描件和模板页数一致。
    """
    scan_pages = get_document_page_count(scan_path)
    template_pages = get_document_page_count(template_path)
    if scan_pages != template_pages:
        raise InputError(
            "input: scan and template page counts must match: "
            f"scan={scan_pages}, template={template_pages}"
        )
    if requested_page is not None:
        if requested_page < 0 or requested_page >= scan_pages:
            raise InputError(f"input: page index out of range: {requested_page}")
        return [requested_page]
    return list(range(scan_pages))


def _prepare_comparable_inputs(
    scan_path: str,
    template_path: str,
    run_dir: Path | None,
) -> tuple[str, str]:
    """将 DOCX 输入自动转为 PDF，确保后续可以统一渲染。"""
    conversion_dir = (run_dir or Path("artifacts") / _new_run_id()) / "_converted"
    return (
        str(_convert_docx_input(scan_path, conversion_dir)),
        str(_convert_docx_input(template_path, conversion_dir)),
    )


def _convert_docx_input(path: str, conversion_dir: Path) -> Path:
    """如果是 DOCX 文件，转为 PDF；否则直接返回原路径。"""
    input_path = Path(path)
    if input_path.suffix.lower() == ".docx":
        return convert_docx_to_pdf(input_path, conversion_dir)
    return input_path


def _report_page_output_paths(
    output_paths: CliOutputPaths,
    page: int,
) -> dict[str, str]:
    """为指定页面生成报告输出路径（page_0001_original.png 等）。"""
    if output_paths.report_dir is None:
        raise InputError("output: report directory is required for page output paths")
    page_number = page + 1
    return {
        "template_output_path": str(
            output_paths.report_dir / f"page_{page_number:04d}_original.png"
        ),
        "candidate_output_path": str(
            output_paths.report_dir / f"page_{page_number:04d}_candidate.png"
        ),
        "ghost_output_path": str(output_paths.report_dir / f"page_{page_number:04d}_heatmap.png"),
    }


def _format_single_result_summary(result: Any, output_paths: CliOutputPaths) -> str:
    """单页模式 CLI 输出摘要。"""
    lines = [f"completed page={result.page + 1} differences={len(result.differences)}"]
    _append_path(lines, "json", output_paths.json)
    _append_path(lines, "visual", output_paths.visual)
    _append_path(lines, "ghost", output_paths.ghost)
    _append_path(lines, "output_dir", output_paths.run_dir)
    return "\n".join(lines)


def _format_report_summary(payload: dict[str, Any], output_paths: CliOutputPaths) -> str:
    """报告模式 CLI 输出摘要。"""
    lines = [
        "completed "
        f"pages={len(payload.get('pages', []))} "
        f"differences={payload.get('total_regions', 0)}"
    ]
    _append_path(lines, "json", output_paths.json)
    _append_path(lines, "html", output_paths.report_html)
    _append_path(lines, "docx", output_paths.report_docx)
    _append_path(lines, "report_dir", output_paths.report_dir)
    _append_path(lines, "output_dir", output_paths.run_dir)
    return "\n".join(lines)


def _append_path(lines: list[str], label: str, path: Path | None) -> None:
    """如果路径非空，追加到输出行列表。"""
    if path is not None:
        lines.append(f"{label}: {path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PixelDiffError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
