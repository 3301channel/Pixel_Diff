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

import os

# 限制底层多线程库，避免多线程并行时线程超订：
# N 个比对线程 × OpenCV/numpy 内部多线程会争抢 CPU，反而变慢。
# 每个线程只保留 1 个内部线程，由线程池并行来吃满 CPU 核心。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

cv2.setNumThreads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pixel_diff import (  # noqa: E402
    AlignmentError,
    ConfigurationError,
    InputError,
    PixelDiffConfig,
    PixelDiffError,
    compare,
)
from pixel_diff._app_paths import resource_path  # noqa: E402
from pixel_diff.io import convert_docx_to_pdf, get_document_page_count, write_json  # noqa: E402
from pixel_diff.logging_setup import setup_logging  # noqa: E402
from pixel_diff.report import (  # noqa: E402
    build_document_report_payload,
    render_html_report,
    write_docx_report,
)


@dataclass(frozen=True)
class CliOutputPaths:
    """单次 CLI 比对运行的输出路径集合。"""

    visual: Path | None
    json: Path | None
    ghost: Path | None
    template: Path | None
    candidate: Path | None
    patch: Path | None
    report_dir: Path | None
    report_html: Path | None
    report_docx: Path | None
    run_dir: Path | None


@dataclass(frozen=True)
class PageCompareJob:
    scan_path: str
    template_path: str
    page: int
    config: PixelDiffConfig
    output_paths: dict[str, str]
    log_file: str | None = None


def _compare_report_page(job: PageCompareJob) -> tuple[int, Any]:
    """Run one report page in a process-safe top-level worker."""
    # 子进程内重新初始化日志（Windows spawn 不继承主进程 handler），
    # 追加写入同一个 run.log，便于统一归档；关闭控制台避免与主进程输出交错。
    if job.log_file is not None:
        setup_logging(log_file=job.log_file, console=False)
    result = compare(
        scan_path=job.scan_path,
        template_path=job.template_path,
        page=job.page,
        config=job.config,
        ghost_output_path=job.output_paths["ghost_output_path"],
        template_output_path=job.output_paths["template_output_path"],
        candidate_output_path=job.output_paths["candidate_output_path"],
        patch_output_dir=job.output_paths.get("patch_output_dir"),
    )
    return job.page, result


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
    parser.add_argument(
        "--export-patches",
        action="store_true",
        help="Explicitly export candidate patches for this run.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run identifier used as the output directory prefix "
             "(default: timestamp). Must match ^[A-Za-z0-9_-]{1,64}$.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override console log level (default: from config, INFO).",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Disable console logging (file logging stays active).",
    )
    parser.add_argument(
        "--report-workers",
        type=int,
        help="Override report_parallel_workers (parallel process count).",
    )
    args = parser.parse_args()

    # 加载配置（YAML 不存在时回退默认）
    config = _load_config(args.config)
    if args.export_patches:
        config = replace(config, patch_export_enabled=True)
    if args.log_level is not None:
        config = replace(config, log_level=args.log_level)
    if args.no_console:
        config = replace(config, log_console=False)
    if args.report_workers is not None:
        config = replace(config, report_parallel_workers=max(1, args.report_workers))

    # 准备输出路径
    if args.run_id is not None:
        _validate_run_id(args.run_id)
    output_paths = _prepare_output_paths(
        scan_path=args.scan_path,
        template_path=args.template_path,
        visual_path=args.visual,
        json_path=args.json,
        ghost_path=args.ghost,
        report_dir=args.report_dir,
        run_id=args.run_id,
    )

    # 初始化日志：优先写到本次运行目录 run.log，否则回退全局汇总日志
    log_file = output_paths.run_dir / "run.log" if output_paths.run_dir else None
    setup_logging(
        log_file=log_file,
        level=config.log_level,
        console=config.log_console,
        enable_file=config.log_file,
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
        wall_started_at = time.perf_counter()
        results = []
        output_paths_by_page: dict[int, dict[str, str]] = {}
        jobs = [
            PageCompareJob(
                scan_path=str(scan_path),
                template_path=str(template_path),
                page=page,
                config=config,
                output_paths=_report_page_output_paths(
                    output_paths,
                    page,
                    include_patches=config.patch_export_enabled,
                ),
                log_file=str(log_file) if log_file is not None else None,
            )
            for page in pages
        ]
        jobs_by_page = {job.page: job for job in jobs}

        def record_page(page: int, result: Any) -> None:
            page_paths = jobs_by_page[page].output_paths
            results.append(result)
            print(_format_page_timing(result, total_pages=len(pages)), flush=True)
            output_paths_by_page[page] = {
                **page_paths,
                "json_output_path": str(output_paths.json),
            }
            if output_paths.report_html is not None:
                output_paths_by_page[page]["html_output_path"] = str(output_paths.report_html)
            if output_paths.report_docx is not None:
                output_paths_by_page[page]["docx_output_path"] = str(output_paths.report_docx)

        if len(jobs) > 1 and config.report_parallel_workers > 1:
            # 用线程池并行（而非进程池）：OpenCV/numpy 的 C 扩展会释放 GIL，
            # 多线程同样能吃到多核；同时避免 PyInstaller 冻结态下进程池的
            # SemLock PermissionError 崩溃。
            with ThreadPoolExecutor(
                max_workers=_effective_report_workers(
                    len(jobs), config.report_parallel_workers
                ),
            ) as executor:
                for page, result in executor.map(_compare_report_page, jobs):
                    record_page(page, result)
        else:
            for job in jobs:
                record_page(*_compare_report_page(job))

        # 构建报告载荷
        report_payload = build_document_report_payload(
            results,
            run_id=output_paths.run_dir.name if output_paths.run_dir else _new_run_id(),
            scan_path=args.scan_path,
            template_path=args.template_path,
            config=config,
            output_paths_by_page=output_paths_by_page,
        )
        report_payload["wall_elapsed_ms"] = int(
            round((time.perf_counter() - wall_started_at) * 1000)
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
        patch_output_dir=output_paths.patch,
    )
    print(_format_single_result_summary(result, output_paths))
    return 0


def _load_config(path: str) -> PixelDiffConfig:
    """Load the requested YAML and never silently replace a missing file.

    先按字面路径查找；若找不到，再尝试相对项目根（冻结态为 exe 同级）
    解析，使打包后用 ``--config sensitive_recall_trial.yaml`` 仍能命中。
    """
    config_path = Path(path)
    if not config_path.is_file():
        alt = resource_path(path)
        if alt.is_file():
            config_path = alt
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {config_path}")
    return PixelDiffConfig.from_yaml(config_path)


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
            template=None, candidate=None, patch=None,
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
            page_report_dir / "page_0001_candidate.jpg"
            if page_report_dir is not None
            else None
        ),
        patch=(
            page_report_dir / "patches" / "page_0001"
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


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_run_id(run_id: str) -> None:
    """校验 --run-id，防止被用于路径穿越或注入非法目录名。"""
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ConfigurationError(
            "output: --run-id must match ^[A-Za-z0-9_-]{1,64}$"
        )


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


def _effective_report_workers(page_count: int, configured_workers: int) -> int:
    """Bound report parallelism by pages and the available CPU count."""
    cpu_cap = os.cpu_count() or 1
    return max(1, min(page_count, configured_workers, cpu_cap))


def _prepare_comparable_inputs(
    scan_path: str,
    template_path: str,
    run_dir: Path | None,
) -> tuple[str, str]:
    """将 DOCX 输入自动转为 PDF，确保后续可以统一渲染。"""
    conversion_dir = (run_dir or Path("artifacts") / _new_run_id()) / "_converted"
    scan_converted = str(_convert_docx_input(scan_path, conversion_dir))
    template_converted = str(_convert_docx_input(template_path, conversion_dir))
    # 图片方向归一化（避免横竖两张图片直接比对）
    rotated_dir = conversion_dir.parent / "_rotated"
    return (
        _normalize_image_orientation(scan_converted, template_converted, rotated_dir),
        template_converted,
    )


def _convert_docx_input(path: str, conversion_dir: Path) -> Path:
    """如果是 DOCX 文件，转为 PDF；否则直接返回原路径。"""
    input_path = Path(path)
    if input_path.suffix.lower() == ".docx":
        return convert_docx_to_pdf(input_path, conversion_dir)
    return input_path


def _is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}


def _normalize_image_orientation(
    scan_path: str, template_path: str, out_dir: Path
) -> str:
    """若 scan 与 template 都是图片且横竖方向不同，旋转 scan 90° 使两者同方向。

    返回旋转后保存路径（已同方向或非图片则返回原 scan_path）。
    旋转后文本方向会变，但几何结构（表格、线框）保持，便于特征对齐。
    """
    if not (_is_image_file(scan_path) and _is_image_file(template_path)):
        return scan_path
    scan = cv2.imread(scan_path)
    template = cv2.imread(template_path)
    if scan is None or template is None:
        return scan_path
    scan_landscape = scan.shape[1] > scan.shape[0]
    template_landscape = template.shape[1] > template.shape[0]
    if scan_landscape == template_landscape:
        return scan_path
    rotated = cv2.rotate(
        scan,
        cv2.ROTATE_90_CLOCKWISE if template_landscape else cv2.ROTATE_90_COUNTERCLOCKWISE,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rotated_{Path(scan_path).name}"
    cv2.imwrite(str(out_path), rotated)
    return str(out_path)


def _report_page_output_paths(
    output_paths: CliOutputPaths,
    page: int,
    *,
    include_patches: bool = True,
) -> dict[str, str]:
    """为指定页面生成报告输出路径（page_0001_original.png 等）。"""
    if output_paths.report_dir is None:
        raise InputError("output: report directory is required for page output paths")
    page_number = page + 1
    paths = {
        "template_output_path": str(
            output_paths.report_dir / f"page_{page_number:04d}_original.jpg"
        ),
        "candidate_output_path": str(
            output_paths.report_dir / f"page_{page_number:04d}_candidate.jpg"
        ),
        "ghost_output_path": str(output_paths.report_dir / f"page_{page_number:04d}_heatmap.png"),
    }
    if include_patches:
        paths["patch_output_dir"] = str(
            output_paths.report_dir / "patches" / f"page_{page_number:04d}"
        )
    return paths


def _format_single_result_summary(result: Any, output_paths: CliOutputPaths) -> str:
    """单页模式 CLI 输出摘要。"""
    lines = [
        _format_page_timing(result, total_pages=1),
        f"completed page={result.page + 1} differences={len(result.differences)}",
    ]
    _append_path(lines, "json", output_paths.json)
    _append_path(lines, "visual", output_paths.visual)
    _append_path(lines, "ghost", output_paths.ghost)
    _append_path(lines, "output_dir", output_paths.run_dir)
    return "\n".join(lines)


def _format_report_summary(payload: dict[str, Any], output_paths: CliOutputPaths) -> str:
    """报告模式 CLI 输出摘要。"""
    pages = payload.get("pages", [])
    algorithm_ms = sum(
        float(page.get("metrics", {}).get("elapsed_ms", page.get("elapsed_ms", 0)))
        for page in pages
    )
    wall_ms = float(payload.get("wall_elapsed_ms", algorithm_ms))
    lines = [
        "completed "
        f"pages={len(pages)} "
        f"differences={payload.get('total_regions', 0)} "
        f"algorithm_time={algorithm_ms / 1000:.2f}s "
        f"wall_time={wall_ms / 1000:.2f}s"
    ]
    _append_path(lines, "json", output_paths.json)
    _append_path(lines, "html", output_paths.report_html)
    _append_path(lines, "docx", output_paths.report_docx)
    _append_path(lines, "report_dir", output_paths.report_dir)
    _append_path(lines, "output_dir", output_paths.run_dir)
    return "\n".join(lines)


def _format_page_timing(result: Any, total_pages: int) -> str:
    """Format the per-page total and available stage durations."""
    metrics = result.metrics
    total_ms = float(metrics.get("elapsed_ms", 0))
    labels = (
        ("render", "render"),
        ("color_filter", "color"),
        ("text_anchor_alignment", "text-anchor"),
        ("global_alignment", "align"),
        ("line_alignment", "line"),
        ("piecewise_alignment", "piecewise"),
        ("local_warp", "warp"),
        ("binarization", "binary"),
        ("difference_text", "diff"),
        ("filtering", "filter"),
        ("risk_review", "risk"),
        ("output", "output"),
    )
    stages = [
        f"{label}={float(metrics[key]) / 1000:.2f}s"
        for stage, label in labels
        if (key := f"timing_{stage}_ms") in metrics
    ]
    suffix = f" {' '.join(stages)}" if stages else ""
    page_number = result.page + 1
    page_label = (
        f"page {page_number}/{total_pages}"
        if total_pages >= page_number
        else f"page {page_number}"
    )
    return (
        f"{page_label} "
        f"differences={len(result.differences)} time={total_ms / 1000:.2f}s{suffix}"
    )


def _append_path(lines: list[str], label: str, path: Path | None) -> None:
    """如果路径非空，追加到输出行列表。"""
    if path is not None:
        lines.append(f"{label}: {path}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AlignmentError:
        # 配准失败（特征点为空/不足、好匹配不足、单应性失败等）统一映射为
        # 中文「文档不一致」文案，避免把英文技术异常透传给 API 调用方。
        # 与 task_service 里 alignment_distorted / inlier / warp_iou 的判定文案风格一致。
        print(
            "两份文档无法配准（特征匹配不足或对齐失败），疑似非同一文档",
            file=sys.stderr,
        )
        raise SystemExit(2)
    except PixelDiffError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
