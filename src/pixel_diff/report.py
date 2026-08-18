"""报告生成：JSON 载荷构建、HTML 渲染和 DOCX 导出。

报告系统输出三个层次的结果：
1. build_report_payload()       — 构建 JSON 可序列化的报告数据结构
2. render_html_report()         — 生成独立的 HTML 可视化报告（含图片引用和差异率）
3. write_docx_report()          — 生成 Word 兼容的 DOCX 报告（纯 XML 组装，无依赖）

差异率公式：
  difference_rate = sum(所有差异区域的轮廓面积) / sum(所有页面的像素面积)
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, cast
from zipfile import ZIP_DEFLATED, ZipFile

from pixel_diff.models import PixelDiffConfig, PixelDiffResult


def build_report_payload(
    result: PixelDiffResult,
    *,
    run_id: str,
    scan_path: str | Path,
    template_path: str | Path,
    config: PixelDiffConfig,
    output_paths: dict[str, str],
) -> dict[str, Any]:
    """为单次比对（单页）构建报告载荷。包装 build_document_report_payload。"""
    return build_document_report_payload(
        [result],
        run_id=run_id,
        scan_path=scan_path,
        template_path=template_path,
        config=config,
        output_paths_by_page={result.page: output_paths},
    )


def build_document_report_payload(
    results: list[PixelDiffResult],
    *,
    run_id: str,
    scan_path: str | Path,
    template_path: str | Path,
    config: PixelDiffConfig,
    output_paths_by_page: dict[int, dict[str, str]],
) -> dict[str, Any]:
    """为一页或多页的比对结果构建完整的 JSON 报告载荷。

    返回的字典包含：
    - 文档级汇总（总页数、总差异数、总差异率）
    - 每页详情（差异区域数、页差异率、配准指标、输出文件路径）
    - 所有差异区域列表（含页码）
    """
    sorted_results = sorted(results, key=lambda item: item.page)

    # 扁平化所有差异区域，附加页码
    regions = [
        {
            "page": result.page + 1,  # 转为 1-based 页码
            **region.to_dict(),
        }
        for result in sorted_results
        for region in result.differences
    ]

    # 逐页统计
    pages = []
    total_diff_area = 0.0
    total_page_area = 0.0
    for result in sorted_results:
        page_area = _page_pixel_area(result.image)
        diff_area = _result_diff_area(result)
        total_page_area += page_area
        total_diff_area += diff_area
        pages.append(
            {
                "page": result.page + 1,
                "page_status": result.status,
                "regions": len(result.differences),
                "image": result.image,
                "diff_area": diff_area,
                "page_area": page_area,
                "difference_rate": _safe_rate(diff_area, page_area),
                "good_matches": result.metrics.get("good_matches"),
                "inlier_ratio": result.metrics.get("inlier_ratio"),
                "feature_detector": result.metrics.get("feature_detector"),
                "feature_detector_fallback": result.metrics.get("feature_detector_fallback"),
                "alignment_feature_downsampled": result.metrics.get(
                    "alignment_feature_downsampled"
                ),
                "alignment_feature_scale": result.metrics.get(
                    "alignment_feature_scale"
                ),
                "alignment_feature_fallback": result.metrics.get(
                    "alignment_feature_fallback"
                ),
                "line_centroid_alignment": result.metrics.get("line_centroid_alignment"),
                "line_centroid_applied": result.metrics.get("line_centroid_applied"),
                "line_centroid_matched_pairs": result.metrics.get(
                    "line_centroid_matched_pairs"
                ),
                "line_centroid_max_abs_offset": result.metrics.get(
                    "line_centroid_max_abs_offset"
                ),
                "line_horizontal_applied": result.metrics.get("line_horizontal_applied"),
                "line_horizontal_anchors": result.metrics.get("line_horizontal_anchors"),
                "line_horizontal_max_abs_offset": result.metrics.get(
                    "line_horizontal_max_abs_offset"
                ),
                "line_affine_applied_lines": result.metrics.get(
                    "line_affine_applied_lines"
                ),
                "line_affine_checked_lines": result.metrics.get(
                    "line_affine_checked_lines"
                ),
                "line_affine_before_iou": result.metrics.get("line_affine_before_iou"),
                "line_affine_after_iou": result.metrics.get("line_affine_after_iou"),
                "line_affine_max_scale_delta": result.metrics.get(
                    "line_affine_max_scale_delta"
                ),
                "line_affine_max_displacement": result.metrics.get(
                    "line_affine_max_displacement"
                ),
                "line_piecewise_applied_lines": result.metrics.get(
                    "line_piecewise_applied_lines"
                ),
                "line_piecewise_checked_lines": result.metrics.get(
                    "line_piecewise_checked_lines"
                ),
                "line_piecewise_anchors": result.metrics.get("line_piecewise_anchors"),
                "line_piecewise_protected_intervals": result.metrics.get(
                    "line_piecewise_protected_intervals"
                ),
                "line_piecewise_before_iou": result.metrics.get(
                    "line_piecewise_before_iou"
                ),
                "line_piecewise_after_iou": result.metrics.get(
                    "line_piecewise_after_iou"
                ),
                "line_piecewise_max_displacement": result.metrics.get(
                    "line_piecewise_max_displacement"
                ),
                "line_piecewise_max_scale_delta": result.metrics.get(
                    "line_piecewise_max_scale_delta"
                ),
                "rigid_text_blocks_attempted": result.metrics.get(
                    "rigid_text_blocks_attempted"
                ),
                "rigid_text_blocks_applied": result.metrics.get(
                    "rigid_text_blocks_applied"
                ),
                "rigid_text_blocks_rejected_overlap": result.metrics.get(
                    "rigid_text_blocks_rejected_overlap"
                ),
                "rigid_text_blocks_rejected_quality": result.metrics.get(
                    "rigid_text_blocks_rejected_quality"
                ),
                "rigid_text_block_applied_lines": result.metrics.get(
                    "rigid_text_block_applied_lines"
                ),
                "residual_line_candidate_lines": result.metrics.get(
                    "residual_line_candidate_lines"
                ),
                "residual_line_attempted_lines": result.metrics.get(
                    "residual_line_attempted_lines"
                ),
                "residual_line_applied_lines": result.metrics.get(
                    "residual_line_applied_lines"
                ),
                "residual_line_before_diff_pixels": result.metrics.get(
                    "residual_line_before_diff_pixels"
                ),
                "residual_line_after_diff_pixels": result.metrics.get(
                    "residual_line_after_diff_pixels"
                ),
                "residual_line_before_long_residuals": result.metrics.get(
                    "residual_line_before_long_residuals"
                ),
                "residual_line_after_long_residuals": result.metrics.get(
                    "residual_line_after_long_residuals"
                ),
                "residual_line_protected_intervals": result.metrics.get(
                    "residual_line_protected_intervals"
                ),
                "residual_line_protected_retention": result.metrics.get(
                    "residual_line_protected_retention"
                ),
                "residual_line_max_displacement": result.metrics.get(
                    "residual_line_max_displacement"
                ),
                "residual_line_max_scale_delta": result.metrics.get(
                    "residual_line_max_scale_delta"
                ),
                "local_warp_enabled": result.metrics.get("local_warp_enabled"),
                "local_warp_applied": result.metrics.get("local_warp_applied"),
                "local_warp_max_displacement": result.metrics.get(
                    "local_warp_max_displacement"
                ),
                "local_warp_mean_displacement": result.metrics.get(
                    "local_warp_mean_displacement"
                ),
                "local_warp_gate_skipped": result.metrics.get(
                    "local_warp_gate_skipped"
                ),
                "local_warp_gate_foreground_iou": result.metrics.get(
                    "local_warp_gate_foreground_iou"
                ),
                "displacement_pairs": result.metrics.get("displacement_pairs"),
                "filter_input_regions": result.metrics.get("filter_input_regions"),
                "filter_level1_removed": result.metrics.get("filter_level1_removed"),
                "filter_level2_removed": result.metrics.get("filter_level2_removed"),
                "filter_level3_annotations": result.metrics.get(
                    "filter_level3_annotations"
                ),
                "filter_ssim_checked": result.metrics.get("filter_ssim_checked"),
                "filter_ssim_max_score": result.metrics.get("filter_ssim_max_score"),
                "risk_review_high": result.metrics.get("risk_review_high"),
                "risk_review_medium": result.metrics.get("risk_review_medium"),
                "risk_review_low": result.metrics.get("risk_review_low"),
                "risk_review_low_filtered": result.metrics.get("risk_review_low_filtered"),
                "risk_review_narrow_stroke_filtered": result.metrics.get(
                    "risk_review_narrow_stroke_filtered"
                ),
                "elapsed_ms": result.metrics.get("elapsed_ms"),
                "timings": {
                    key: value
                    for key, value in result.metrics.items()
                    if key.startswith("timing_")
                },
                "outputs": output_paths_by_page.get(result.page, {}),
            }
        )

    return {
        "run_id": run_id,
        "template_path": str(template_path),
        "scan_path": str(scan_path),
        "dpi": config.dpi,
        "status": (
            "completed"
            if all(result.status == "completed" for result in sorted_results)
            else "partial"
        ),
        "total_pages": len(sorted_results),
        "total_regions": len(regions),
        "total_diff_area": total_diff_area,
        "total_page_area": total_page_area,
        "difference_rate": _safe_rate(total_diff_area, total_page_area),
        "pages": pages,
        "regions": regions,
        "metrics": {
            "elapsed_ms": sum(
                int(result.metrics.get("elapsed_ms", 0)) for result in sorted_results
            ),
            "pages": len(sorted_results),
        },
        "outputs": {
            "json_output_path": _first_output_path(output_paths_by_page, "json_output_path"),
            "html_output_path": _first_output_path(output_paths_by_page, "html_output_path"),
            "docx_output_path": _first_output_path(output_paths_by_page, "docx_output_path"),
        },
    }


def render_html_report(
    payload: dict[str, Any],
    *,
    report_dir: str | Path,
) -> str:
    """将报告载荷渲染为独立的 HTML 字符串。

    HTML 报告包含：
    - 文档差异率（红色卡片醒目展示）
    - 汇总卡片（疑似差异区域数、比对页数、渲染 DPI、总耗时）
    - 比对信息（模板文件名、扫描件文件名、运行编号）
    - 逐页展示：配准指标 + 图片行（原始/待检/比对结果）+ 差异坐标表
    - 支持响应式布局（移动端适配）

    图片通过相对路径引用 report_dir 下的文件。
    """
    report_path = Path(report_dir)
    pages = cast(list[dict[str, Any]], payload.get("pages", []))
    regions = cast(list[dict[str, Any]], payload.get("regions", []))
    regions_by_page = _group_regions_by_page(regions)

    # 渲染所有页面的 HTML 片段
    page_sections = "\n".join(
        _render_page_section(page, report_path, regions_by_page) for page in pages
    )

    # 安全转义摘要字段
    run_id = escape(str(payload.get("run_id", "")))
    template_name = escape(Path(str(payload.get("template_path", ""))).name)
    scan_name = escape(Path(str(payload.get("scan_path", ""))).name)
    total_regions = escape(str(payload.get("total_regions", 0)))
    total_pages = escape(str(payload.get("total_pages", len(pages))))
    dpi = escape(str(payload.get("dpi", "")))
    difference_rate = escape(_format_percent(payload.get("difference_rate", 0)))
    metrics = cast(dict[str, Any], payload.get("metrics", {}))
    elapsed_ms = escape(str(metrics.get("elapsed_ms", "")))
    docx_href = _docx_href(payload, report_path)

    # 内联 CSS + HTML 模板（中文本地化）
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pixel-Diff 像素级差异检测报告</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background: #f3f5f7;
      color: #20242a;
      font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif;
    }}
    .container {{
      max-width: 1800px;
      margin: 0 auto;
      background: #fff;
      border: 1px solid #dfe3e8;
      border-radius: 8px;
      padding: 28px;
    }}
    .report-header {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 18px;
      border-bottom: 3px solid #1f6feb;
      padding-bottom: 14px;
      margin-bottom: 18px;
    }}
    h1 {{ margin: 0; font-size: 28px; }}
    h2 {{ margin: 28px 0 14px; font-size: 20px; }}
    .export-button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 150px;
      padding: 10px 18px;
      border-radius: 4px;
      background: #1683f5;
      color: #fff;
      font-weight: 700;
      text-decoration: none;
      white-space: nowrap;
    }}
    .difference-banner {{
      display: grid;
      grid-template-columns: minmax(220px, 360px) 1fr;
      gap: 18px;
      align-items: stretch;
      margin: 18px 0;
    }}
    .difference-rate-card {{
      border: 2px solid #d92d20;
      background: #fff4f2;
      border-radius: 8px;
      padding: 18px;
    }}
    .difference-rate-card .value {{
      font-size: 40px;
      line-height: 1;
      font-weight: 800;
      color: #d92d20;
    }}
    .difference-rate-card .label {{
      margin-top: 8px;
      color: #7a271a;
      font-weight: 700;
    }}
    .difference-note {{
      display: flex;
      align-items: center;
      color: #667085;
      line-height: 1.7;
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 14px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 14px;
      margin: 18px 0;
    }}
    .summary-card {{
      border-left: 4px solid #1f6feb;
      background: #f8fafc;
      padding: 14px;
      border-radius: 6px;
    }}
    .summary-card .value {{
      font-size: 26px;
      font-weight: 700;
      color: #1f6feb;
    }}
    .summary-card .label {{ color: #667085; font-size: 13px; }}
    .meta {{
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 8px 14px;
      color: #3b414a;
    }}
    .meta strong {{ color: #1f2937; }}
    .page-section {{
      background: #fafafa;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 20px;
      margin: 24px 0;
    }}
    .image-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      margin-top: 12px;
    }}
    .image-col p {{
      margin: 0 0 8px;
      text-align: center;
      font-weight: 600;
      color: #3b414a;
    }}
    .image-col img {{
      width: 100%;
      border: 1px solid #d0d5dd;
      border-radius: 4px;
      background: #fff;
    }}
    .region-details {{ margin-top: 18px; }}
    .region-details h3 {{ margin: 0 0 10px; font-size: 16px; color: #3b414a; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
    }}
    th {{ background: #f2f4f7; font-weight: 700; }}
    .footer {{
      margin-top: 24px;
      padding-top: 14px;
      border-top: 1px solid #e5e7eb;
      color: #667085;
      font-size: 12px;
      text-align: center;
    }}
    @media (max-width: 760px) {{
      .report-header, .difference-banner {{ display: block; }}
      .export-button {{ margin-top: 14px; width: 100%; }}
      .difference-note {{ margin-top: 12px; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="report-header">
      <h1>Pixel-Diff 像素级差异检测报告</h1>
      <a class="export-button" href="{docx_href}" download>导出比对报告</a>
    </div>
    <div class="difference-banner">
      <div class="difference-rate-card">
        <div class="value">{difference_rate}</div>
        <div class="label">文档差异率</div>
      </div>
      <div class="difference-note">
        差异率按"疑似差异轮廓面积总和 / 页面像素面积总和"计算，用于快速评估页面变化幅度；
        它不代表语义改动比例，也不替代人工复核结论。
      </div>
    </div>
    <div class="summary">
      <div class="summary-card">
        <div class="value">{total_regions}</div>
        <div class="label">疑似差异区域</div>
      </div>
      <div class="summary-card">
        <div class="value">{total_pages}</div>
        <div class="label">比对页数</div>
      </div>
      <div class="summary-card">
        <div class="value">{dpi}</div>
        <div class="label">渲染 DPI</div>
      </div>
      <div class="summary-card">
        <div class="value">{elapsed_ms}</div>
        <div class="label">总耗时 ms</div>
      </div>
    </div>
    <h2>比对信息</h2>
    <div class="meta">
      <strong>模板/审批通过文件</strong><span>{template_name}</span>
      <strong>待检/扫描文件</strong><span>{scan_name}</span>
      <strong>运行编号</strong><span>{run_id}</span>
    </div>
{page_sections}
    <div class="footer">
      Pixel-Diff 仅提供像素级疑似差异供人工复核，不直接给出法律、合规或合同效力结论。
    </div>
  </div>
</body>
</html>
"""


def write_docx_report(payload: dict[str, Any], output_path: str | Path) -> None:
    """将报告载荷写入 Word 兼容的 DOCX 文件。

    DOCX 本质是 ZIP 包，内含 XML 文件。此方法直接组装：
    - [Content_Types].xml — 内容类型声明
    - _rels/.rels         — 根关系文件
    - word/document.xml   — 文档内容（含标题、表格、坐标）
    - word/styles.xml     — 样式定义（Title、Heading1、Heading2）
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _docx_content_types())
        archive.writestr("_rels/.rels", _docx_root_relationships())
        archive.writestr("word/document.xml", _render_docx_document(payload))
        archive.writestr("word/styles.xml", _docx_styles())


# ═══════════════════════════════════════════════════════════════════
# HTML 渲染辅助函数
# ═══════════════════════════════════════════════════════════════════

def _render_image_blocks(outputs: dict[str, Any], report_dir: Path) -> str:
    """渲染页面图片行（原始文档、待检文档、比对结果）的 HTML。"""
    blocks = [
        ("template_output_path", "原始文档（标准化）"),
        ("candidate_output_path", "待检文档（原始）"),
        ("ghost_output_path", "比对结果"),
    ]
    rendered: list[str] = []
    for key, label in blocks:
        raw_path = outputs.get(key)
        if not raw_path:
            continue
        relative = _relative_path_for_html(raw_path, report_dir)
        rendered.append(
            f"""      <div class="image-col">
        <p>{escape(label)}</p>
        <a href="{relative}" target="_blank"><img src="{relative}" alt="{escape(label)}"></a>
      </div>"""
        )
    return "\n".join(rendered)


def _render_page_section(
    page: dict[str, Any],
    report_dir: Path,
    regions_by_page: dict[int, list[dict[str, Any]]],
) -> str:
    """渲染单页详情的 HTML 片段，含配准指标、图片行、差异坐标表。"""
    page_number = escape(str(page.get("page", "")))
    page_index = _coerce_int(page.get("page"), default=0)
    regions = escape(str(page.get("regions", 0)))
    difference_rate = escape(_format_percent(page.get("difference_rate", 0)))
    good_matches = escape(str(page.get("good_matches", "")))
    inlier_ratio = page.get("inlier_ratio", "")
    if isinstance(inlier_ratio, float):
        inlier_ratio_text = f"{inlier_ratio:.4f}"
    else:
        inlier_ratio_text = str(inlier_ratio)
    feature_detector = escape(str(page.get("feature_detector", "")))
    feature_detector_fallback = "是" if page.get("feature_detector_fallback") else "否"
    line_centroid_alignment = "是" if page.get("line_centroid_alignment") else "否"
    line_centroid_applied = "是" if page.get("line_centroid_applied") else "否"
    line_centroid_matched_pairs = escape(str(page.get("line_centroid_matched_pairs", "")))
    line_centroid_max_abs_offset = escape(str(page.get("line_centroid_max_abs_offset", "")))
    filter_level1_removed = escape(str(page.get("filter_level1_removed", "")))
    filter_level2_removed = escape(str(page.get("filter_level2_removed", "")))
    filter_level3_annotations = escape(str(page.get("filter_level3_annotations", "")))
    risk_review_high = escape(str(page.get("risk_review_high", "")))
    risk_review_medium = escape(str(page.get("risk_review_medium", "")))
    risk_review_low = escape(str(page.get("risk_review_low", "")))
    risk_review_low_filtered = escape(str(page.get("risk_review_low_filtered", "")))
    filter_ssim_checked = escape(str(page.get("filter_ssim_checked", "")))
    filter_ssim_max_score = page.get("filter_ssim_max_score", "")
    if isinstance(filter_ssim_max_score, float):
        filter_ssim_max_score_text = f"{filter_ssim_max_score:.4f}"
    else:
        filter_ssim_max_score_text = str(filter_ssim_max_score)
    elapsed_ms = escape(str(page.get("elapsed_ms", "")))
    outputs = cast(dict[str, Any], page.get("outputs", {}))
    image_blocks = _render_image_blocks(outputs, report_dir)
    region_rows = "\n".join(
        _render_page_region_row(region) for region in regions_by_page.get(page_index, [])
    )
    if not region_rows:
        region_rows = '<tr><td colspan="13">本页未发现疑似差异区域。</td></tr>'
    return f"""    <div class="page-section">
      <h2>第 {page_number} 页</h2>
    <div class="meta">
      <strong>疑似差异区域</strong><span>{regions}</span>
      <strong>本页差异率</strong><span>{difference_rate}</span>
      <strong>有效匹配点</strong><span>{good_matches}</span>
      <strong>内点比例</strong><span>{escape(inlier_ratio_text)}</span>
      <strong>特征算法</strong><span>{feature_detector}</span>
      <strong>发生回退</strong><span>{feature_detector_fallback}</span>
      <strong>行质心补偿</strong><span>{line_centroid_alignment}</span>
      <strong>补偿生效</strong><span>{line_centroid_applied}</span>
      <strong>行匹配数</strong><span>{line_centroid_matched_pairs}</span>
      <strong>最大行偏移</strong><span>{line_centroid_max_abs_offset}</span>
      <strong>Level1过滤</strong><span>{filter_level1_removed}</span>
      <strong>Level2过滤</strong><span>{filter_level2_removed}</span>
      <strong>SSIM检查</strong><span>{filter_ssim_checked}</span>
      <strong>SSIM最高分</strong><span>{escape(filter_ssim_max_score_text)}</span>
      <strong>文本注释</strong><span>{filter_level3_annotations}</span>
      <strong>高风险</strong><span>{risk_review_high}</span>
      <strong>中风险</strong><span>{risk_review_medium}</span>
      <strong>低风险</strong><span>{risk_review_low}</span>
      <strong>低风险过滤</strong><span>{risk_review_low_filtered}</span>
      <strong>耗时 ms</strong><span>{elapsed_ms}</span>
    </div>
    <div class="image-row">
{image_blocks}
    </div>
    <div class="region-details">
      <h3>本页差异坐标</h3>
      <table>
        <thead>
          <tr>
            <th>#</th><th>差异类型</th><th>分类依据</th><th>新增像素</th><th>删除像素</th><th>分类置信度</th><th>风险</th><th>原因</th><th>模板文本</th>
            <th>OCR文本</th><th>位置 (x,y)</th><th>尺寸 (w×h)</th><th>面积</th>
          </tr>
        </thead>
        <tbody>
{region_rows}
        </tbody>
      </table>
    </div>
    </div>"""


def _render_page_region_row(region: dict[str, Any]) -> str:
    """渲染单个差异区域的 HTML 表行。"""
    region_id = escape(str(region.get("id", "")))
    x = escape(str(region.get("x", "")))
    y = escape(str(region.get("y", "")))
    width = escape(str(region.get("width", "")))
    height = escape(str(region.get("height", "")))
    area = escape(str(region.get("area", "")))
    risk_level = escape(str(region.get("risk_level", "")))
    risk_reason = escape(str(region.get("risk_reason", "")))
    template_text = escape(str(region.get("template_text", "")))
    ocr_text = escape(str(region.get("ocr_text", "")))
    change_label = escape(str(region.get("change_label", "")))
    classification_reason = escape(str(region.get("classification_reason", "")))
    added_pixels = escape(str(region.get("added_pixels", "")))
    deleted_pixels = escape(str(region.get("deleted_pixels", "")))
    confidence_value = region.get("classification_confidence", "")
    confidence = (
        f"{float(confidence_value):.3f}" if confidence_value not in ("", None) else ""
    )
    return f"""        <tr>
          <td>{region_id}</td>
          <td>{change_label}</td>
          <td>{classification_reason}</td>
          <td>{added_pixels}</td>
          <td>{deleted_pixels}</td>
          <td>{confidence}</td>
          <td>{risk_level}</td>
          <td>{risk_reason}</td>
          <td>{template_text}</td>
          <td>{ocr_text}</td>
          <td>({x}, {y})</td>
          <td>{width} × {height}</td>
          <td>{area}</td>
        </tr>"""


# ═══════════════════════════════════════════════════════════════════
# DOCX 生成辅助函数（纯 XML 组装，无第三方依赖）
# ═══════════════════════════════════════════════════════════════════

def _render_docx_document(payload: dict[str, Any]) -> str:
    """渲染 DOCX 文档主体 XML（word/document.xml）。"""
    pages = cast(list[dict[str, Any]], payload.get("pages", []))
    regions = cast(list[dict[str, Any]], payload.get("regions", []))
    regions_by_page = _group_regions_by_page(regions)

    body = [
        _docx_paragraph("Pixel-Diff 像素级差异检测报告", style="Title"),
        _docx_paragraph(
            f"文档差异率：{_format_percent(payload.get('difference_rate', 0))}",
            bold=True,
        ),
        _docx_paragraph(
            "差异率按\u201c疑似差异轮廓面积总和 / 页面像素面积总和\u201d计算，"
            "用于快速评估页面变化幅度，不代表语义改动比例。"
        ),
        _docx_table(
            [
                ("模板/审批通过文件", Path(str(payload.get("template_path", ""))).name),
                ("待检/扫描文件", Path(str(payload.get("scan_path", ""))).name),
                ("运行编号", str(payload.get("run_id", ""))),
                ("比对页数", str(payload.get("total_pages", ""))),
                ("疑似差异区域", str(payload.get("total_regions", ""))),
                ("渲染 DPI", str(payload.get("dpi", ""))),
            ]
        ),
    ]

    # 逐页渲染
    for page in pages:
        page_number = _coerce_int(page.get("page"), default=0)
        body.append(_docx_paragraph(f"第 {page_number} 页", style="Heading1"))
        body.append(
            _docx_table(
                [
                    ("疑似差异区域", str(page.get("regions", ""))),
                    ("本页差异率", _format_percent(page.get("difference_rate", 0))),
                    ("有效匹配点", str(page.get("good_matches", ""))),
                    ("内点比例", _format_ratio(page.get("inlier_ratio", ""))),
                    ("特征算法", str(page.get("feature_detector", ""))),
                    ("发生回退", "是" if page.get("feature_detector_fallback") else "否"),
                    ("行质心补偿", "是" if page.get("line_centroid_alignment") else "否"),
                    ("补偿生效", "是" if page.get("line_centroid_applied") else "否"),
                    ("行匹配数", str(page.get("line_centroid_matched_pairs", ""))),
                    ("最大行偏移", str(page.get("line_centroid_max_abs_offset", ""))),
                    ("Level1过滤", str(page.get("filter_level1_removed", ""))),
                    ("Level2过滤", str(page.get("filter_level2_removed", ""))),
                    ("SSIM检查", str(page.get("filter_ssim_checked", ""))),
                    ("SSIM最高分", _format_ratio(page.get("filter_ssim_max_score", ""))),
                    ("文本注释", str(page.get("filter_level3_annotations", ""))),
                    ("高风险", str(page.get("risk_review_high", ""))),
                    ("中风险", str(page.get("risk_review_medium", ""))),
                    ("低风险", str(page.get("risk_review_low", ""))),
                    ("低风险过滤", str(page.get("risk_review_low_filtered", ""))),
                    ("耗时 ms", str(page.get("elapsed_ms", ""))),
                ]
            )
        )
        page_regions = regions_by_page.get(page_number, [])
        body.append(_docx_paragraph("本页差异坐标", style="Heading2"))
        if page_regions:
            table_rows: list[tuple[Any, ...]] = [
                (
                    "#",
                    "差异类型",
                    "分类依据",
                    "新增像素",
                    "删除像素",
                    "分类置信度",
                    "风险",
                    "原因",
                    "模板文本",
                    "OCR文本",
                    "位置 (x,y)",
                    "尺寸",
                    "面积",
                )
            ]
            table_rows.extend(_docx_region_row(region) for region in page_regions)
            body.append(_docx_table(table_rows))
        else:
            body.append(_docx_paragraph("本页未发现疑似差异区域。"))

    body.append(
        _docx_paragraph(
            "Pixel-Diff 仅提供像素级疑似差异供人工复核，不直接给出法律、合规或合同效力结论。"
        )
    )

    # 组装完整 document.xml
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body)
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>'
        "</w:sectPr></w:body></w:document>"
    )


def _docx_region_row(region: dict[str, Any]) -> tuple[str, ...]:
    """将差异区域转为 DOCX 表格行元组。"""
    x = str(region.get("x", ""))
    y = str(region.get("y", ""))
    width = str(region.get("width", ""))
    height = str(region.get("height", ""))
    confidence = region.get("classification_confidence", "")
    return (
        str(region.get("id", "")),
        str(region.get("change_label", "")),
        str(region.get("classification_reason", "")),
        str(region.get("added_pixels", "")),
        str(region.get("deleted_pixels", "")),
        f"{float(confidence):.3f}" if confidence not in ("", None) else "",
        str(region.get("risk_level", "")),
        str(region.get("risk_reason", "")),
        str(region.get("template_text", "")),
        str(region.get("ocr_text", "")),
        f"({x}, {y})",
        f"{width} × {height}",
        str(region.get("area", "")),
    )


def _docx_paragraph(text: str, *, style: str | None = None, bold: bool = False) -> str:
    """生成 DOCX 段落 XML。"""
    style_xml = f'<w:pPr><w:pStyle w:val="{_xml_escape(style)}"/></w:pPr>' if style else ""
    bold_xml = "<w:b/>" if bold else ""
    return (
        "<w:p>"
        f"{style_xml}<w:r><w:rPr>{bold_xml}</w:rPr>"
        f'<w:t xml:space="preserve">{_xml_escape(text)}</w:t></w:r></w:p>'
    )


def _docx_table(rows: list[tuple[Any, ...]]) -> str:
    """生成 DOCX 带边框表格 XML。"""
    row_xml = []
    for row in rows:
        cells = "".join(
            "<w:tc><w:tcPr><w:tcW w:w=\"2400\" w:type=\"dxa\"/></w:tcPr>"
            f"{_docx_paragraph(str(cell))}</w:tc>"
            for cell in row
        )
        row_xml.append(f"<w:tr>{cells}</w:tr>")
    return (
        "<w:tbl><w:tblPr><w:tblW w:w=\"0\" w:type=\"auto\"/>"
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="D0D5DD"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="D0D5DD"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="D0D5DD"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="D0D5DD"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D0D5DD"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="D0D5DD"/>'
        "</w:tblBorders></w:tblPr>"
        + "".join(row_xml)
        + "</w:tbl>"
    )


# ═══════════════════════════════════════════════════════════════════
# DOCX 固定部件（[Content_Types].xml, _rels/.rels, styles.xml）
# ═══════════════════════════════════════════════════════════════════

def _docx_content_types() -> str:
    """DOCX [Content_Types].xml 固定内容。"""
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""


def _docx_root_relationships() -> str:
    """DOCX _rels/.rels 固定内容。"""
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>"""


def _docx_styles() -> str:
    """DOCX word/styles.xml：定义 Title / Heading1 / Heading2 样式。"""
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:rPr><w:b/><w:sz w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:rPr><w:b/><w:sz w:val="28"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:rPr><w:b/><w:sz w:val="24"/></w:rPr>
  </w:style>
</w:styles>"""


# ═══════════════════════════════════════════════════════════════════
# 通用工具函数
# ═══════════════════════════════════════════════════════════════════

def _group_regions_by_page(regions: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """按页码（1-based）对差异区域分组。"""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for region in regions:
        page = _coerce_int(region.get("page"), default=0)
        grouped.setdefault(page, []).append(region)
    return grouped


def _coerce_int(value: Any, *, default: int) -> int:
    """安全地将值转为 int，失败时返回默认值。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _relative_path_for_html(path: Any, report_dir: Path) -> str:
    """将绝对/相对路径转换为相对于 report_dir 的 HTML 可用路径。"""
    raw_path = Path(str(path))
    try:
        relative = raw_path.resolve().relative_to(report_dir.resolve())
    except ValueError:
        relative = Path("..") / raw_path.name
    return escape(relative.as_posix())


def _docx_href(payload: dict[str, Any], report_dir: Path) -> str:
    """HTML 报告中 DOCX 下载链接的路径。"""
    outputs = cast(dict[str, Any], payload.get("outputs", {}))
    docx_path = outputs.get("docx_output_path")
    if docx_path:
        return _relative_path_for_html(docx_path, report_dir)
    return "diff_report.docx"


def _first_output_path(
    output_paths_by_page: dict[int, dict[str, str]],
    key: str,
) -> str | None:
    """从分页输出路径中取第一个存在的。按页码排序查找。"""
    for page in sorted(output_paths_by_page):
        value = output_paths_by_page[page].get(key)
        if value is not None:
            return value
    return None


def _page_pixel_area(image: dict[str, Any]) -> float:
    """计算页面像素总面积 = width × height。"""
    try:
        width = float(image.get("width", 0))
        height = float(image.get("height", 0))
    except (TypeError, ValueError):
        return 0.0
    return max(width, 0.0) * max(height, 0.0)


def _result_diff_area(result: PixelDiffResult) -> float:
    """计算单页所有差异区域的轮廓面积总和。"""
    return sum(max(float(region.area), 0.0) for region in result.differences)


def _safe_rate(diff_area: float, page_area: float) -> float:
    """安全除法：差异面积 / 页面面积，分母为 0 时返回 0。"""
    if page_area <= 0:
        return 0.0
    return diff_area / page_area


def _format_percent(value: Any) -> str:
    """将比率格式化为百分比字符串，如 "0.1234%". """
    try:
        rate = float(value)
    except (TypeError, ValueError):
        rate = 0.0
    return f"{rate * 100:.4f}%"


def _format_ratio(value: Any) -> str:
    """格式化浮点比率为 4 位小数字符串。"""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _xml_escape(value: str | None) -> str:
    """XML 特殊字符转义（< > & " '）。"""
    if value is None:
        return ""
    return escape(value, quote=True)
