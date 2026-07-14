"""管道编排 —— PixelDiffEngine 串联所有算法步骤。

compare() 方法是整个管道的入口，按顺序执行：
  1. 加载文档页为 BGR → 2. 颜色过滤去红章蓝签 → 3. SIFT 配准对齐
  → 4. 二值化 → 5. XOR 差异检测 + 边缘裁剪 + 形态学清理
  → 6. 轮廓提取 → 7. 局部相似性过滤 → 8. 可视化输出 + 结果构造

同时提供模块级 compare() 便捷函数，无需手动实例化引擎。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pixel_diff.alignment import align_scan_to_template_bgr
from pixel_diff.binarization import binarize_scan_bgr, binarize_template_bgr
from pixel_diff.color_filter import remove_colored_marks_bgr
from pixel_diff.differ import crop_edges, xor_difference
from pixel_diff.io import (
    load_document_page_bgr,
    validate_image_array,
    write_image_bgr,
    write_json,
)
from pixel_diff.models import PixelDiffConfig, PixelDiffResult
from pixel_diff.morphology import clean_difference_mask
from pixel_diff.regions import extract_regions, filter_locally_similar_regions
from pixel_diff.visualization import draw_regions, draw_text_ghost_comparison

logger = logging.getLogger(__name__)


class PixelDiffEngine:
    """像素级差异比对管道编排器。不包含算法细节，仅负责串联各步骤。"""

    def __init__(self, config: PixelDiffConfig | None = None) -> None:
        """初始化引擎。

        Args:
            config: 算法配置。未提供时使用 PixelDiffConfig() 默认值。
        """
        self.config = config or PixelDiffConfig()
        self.config.validate()

    def compare(
        self,
        scan_path: str | Path,
        template_path: str | Path,
        page: int = 0,
        visual_output_path: str | Path | None = None,
        json_output_path: str | Path | None = None,
        ghost_output_path: str | Path | None = None,
        template_output_path: str | Path | None = None,
        candidate_output_path: str | Path | None = None,
    ) -> PixelDiffResult:
        """将一页待检文件（扫描件）与模板文件做像素级比对。

        Args:
            scan_path:     待检文件路径（PDF/图片/DOCX）
            template_path: 审批通过模板文件路径
            page:          页面索引（0-based），默认首页
            visual_output_path:    红框标注图输出路径（可选）
            json_output_path:      结果 JSON 输出路径（可选）
            ghost_output_path:     文字残影图输出路径（可选）
            template_output_path:  模板原图输出路径（可选，调试用）
            candidate_output_path: 对齐后扫描件输出路径（可选，调试用）

        Returns:
            PixelDiffResult: 包含差异列表、配准指标、耗时的完整结果。
        """
        # ── 阶段1：输入 ──
        start = time.perf_counter()
        logger.info("pixel_diff task started")

        # 渲染文档页为 BGR uint8（300 DPI）
        template_bgr = load_document_page_bgr(template_path, page, self.config.dpi)
        scan_bgr = load_document_page_bgr(scan_path, page, self.config.dpi)
        validate_image_array(template_bgr, "template")
        validate_image_array(scan_bgr, "scan")

        # 记录模板尺寸，后续所有差异坐标以此为基准
        template_height, template_width = template_bgr.shape[:2]
        self.config.validate_for_image(template_width, template_height)

        # ── 阶段2：预处理 ──
        # 2a. 颜色过滤：去除扫描件上的红色公章和蓝色手写签名
        filtered_scan = remove_colored_marks_bgr(scan_bgr, self.config)

        # 2b. SIFT/FLANN/RANSAC 全局配准：将扫描件对齐到模板坐标系
        alignment = align_scan_to_template_bgr(filtered_scan, template_bgr, self.config)

        # 可选：输出调试用中间图像
        metadata: dict[str, str] = {}
        if template_output_path is not None:
            metadata["template_output_path"] = write_image_bgr(template_output_path, template_bgr)
        if candidate_output_path is not None:
            metadata["candidate_output_path"] = write_image_bgr(
                candidate_output_path,
                alignment.aligned_bgr,
            )

        # ── 阶段3：二值化 ──
        # 扫描件用自适应阈值（应对光照不均），模板用 Otsu
        scan_binary = binarize_scan_bgr(alignment.aligned_bgr, self.config)
        template_binary = binarize_template_bgr(template_bgr)

        # ── 阶段4：差异检测 ──
        # 4a. XOR 异或：暴露所有像素级差异
        diff = xor_difference(scan_binary, template_binary)
        # 4b. 边缘裁剪：抑制扫描仪边框伪影（40px）
        diff = crop_edges(diff, self.config.crop_margin)
        # 4c. 形态学清理：去噪 + 碎片合并
        diff = clean_difference_mask(diff, self.config)

        # ── 阶段5：区域分析 ──
        # 5a. 轮廓提取：从差异掩码提取外轮廓 → DifferenceRegion 列表
        regions = extract_regions(diff, self.config.min_diff_area)
        # 5b. 局部相似性过滤：7 级策略剔除配准残余假阳性
        regions = filter_locally_similar_regions(
            regions,
            scan_binary=scan_binary,
            template_binary=template_binary,
            config=self.config,
        )

        # ── 阶段6：可视化输出 ──
        visual_path: str | None = None
        if regions and visual_output_path is not None:
            visual = draw_regions(template_bgr, regions)
            visual_path = write_image_bgr(visual_output_path, visual)

        if ghost_output_path is not None:
            ghost = draw_text_ghost_comparison(template_binary, scan_binary)
            if regions:
                ghost = draw_regions(ghost, regions)
            metadata["ghost_output_path"] = write_image_bgr(ghost_output_path, ghost)

        # ── 构建结果 ──
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))
        result = PixelDiffResult(
            status="completed",
            page=page,
            image={"width": template_width, "height": template_height, "dpi": self.config.dpi},
            differences=regions,
            metrics={
                "elapsed_ms": elapsed_ms,
                "good_matches": alignment.good_matches,
                "inlier_ratio": alignment.inlier_ratio,
            },
            visual_output_path=visual_path,
            metadata=metadata,
        )

        # 可选：输出 JSON 结果
        if json_output_path is not None:
            write_json(json_output_path, result.to_dict())

        logger.info("pixel_diff task completed regions=%s elapsed_ms=%s", len(regions), elapsed_ms)
        return result


def compare(
    scan_path: str | Path,
    template_path: str | Path,
    page: int = 0,
    config: PixelDiffConfig | None = None,
    visual_output_path: str | Path | None = None,
    json_output_path: str | Path | None = None,
    ghost_output_path: str | Path | None = None,
    template_output_path: str | Path | None = None,
    candidate_output_path: str | Path | None = None,
) -> PixelDiffResult:
    """便捷函数：无需手动实例化引擎即可比对一对扫描件/模板。

    相当于 PixelDiffEngine(config).compare(...) 的快捷方式。
    """
    return PixelDiffEngine(config).compare(
        scan_path=scan_path,
        template_path=template_path,
        page=page,
        visual_output_path=visual_output_path,
        json_output_path=json_output_path,
        ghost_output_path=ghost_output_path,
        template_output_path=template_output_path,
        candidate_output_path=candidate_output_path,
    )
