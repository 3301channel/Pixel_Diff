"""管道编排 —— PixelDiffEngine 串联所有算法步骤。

compare() 方法是整个管道的入口，按顺序执行：
  1. 加载文档页为 BGR → 2. 颜色过滤去红章蓝签 → 3. SURF 配准对齐
  → 4. 二值化 → 5. XOR 差异检测 + 边缘裁剪 + 形态学清理
  → 6. 轮廓提取 → 7. 局部相似性过滤 → 8. 可视化输出 + 结果构造

同时提供模块级 compare() 便捷函数，无需手动实例化引擎。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pixel_diff.alignment import align_scan_to_template_bgr, _is_homography_distorted
from pixel_diff.binarization import binarize_scan_bgr, binarize_template_bgr
from pixel_diff.change_classification import classify_difference_regions
from pixel_diff.color_filter import remove_colored_marks_bgr
from pixel_diff.differ import crop_edges, xor_difference
from pixel_diff.displacement import pair_displaced_regions
from pixel_diff.filter_pipeline import apply_multilevel_filters
from pixel_diff.io import (
    load_document_page_bgr,
    validate_image_array,
    write_image_bgr,
    write_json,
)
from pixel_diff.line_affine_alignment import align_text_lines_affine_bgr
from pixel_diff.line_alignment import align_text_lines_by_centroid_bgr
from pixel_diff.line_piecewise_alignment import align_text_lines_piecewise_bgr
from pixel_diff.local_warp import apply_constrained_local_warp_bgr
from pixel_diff.logging_setup import setup_logging
from pixel_diff.models import PixelDiffConfig, PixelDiffResult
from pixel_diff.morphology import prepare_difference_masks
from pixel_diff.patch_export import export_region_patches
from pixel_diff.regions import extract_regions, filter_locally_similar_regions
from pixel_diff.residual_line_alignment import realign_residual_text_lines_bgr
from pixel_diff.risk_review import apply_risk_review
from pixel_diff.text_anchor_alignment import (
    align_by_text_anchors_bgr,
    extract_ocr_text_anchor_lines,
)
from pixel_diff.text_layer import (
    build_pdf_image_keep_mask,
    build_unchanged_text_mask,
    extract_sensitive_text_recall_regions,
    extract_template_text_by_region,
    extract_text_anchor_lines,
    extract_text_difference_regions,
    filter_recalled_similarity_regions,
    has_pdf_text_layer,
    merge_regions,
)
from pixel_diff.timing import StageTimer
from pixel_diff.visualization import draw_regions, draw_text_ghost_comparison

logger = logging.getLogger(__name__)


def _ensure_logging(config: PixelDiffConfig) -> None:
    """首次进入比对时若尚无 handler，则按配置初始化日志。

    覆盖 Python API 直连调用（不经过 CLI）的场景：此时 ``pixel_diff``
    logger 未配置任何 handler，默认日志无处落地。CLI 已在主入口配置好
    handler，本函数检测到已有 handler 时会直接跳过，避免重复配置。
    """
    if logging.getLogger("pixel_diff").handlers:
        return
    setup_logging(
        level=config.log_level,
        console=config.log_console,
        enable_file=config.log_file,
    )


def _apply_orientation(
    template_bgr_raw: np.ndarray,
    scan_bgr_raw: np.ndarray,
    rotation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """对模板和扫描件统一应用方向归一化（横版→竖版）。

    不同扫描件横放方向可能不同（顺时针/逆时针 90° 横放），错误的旋转
    会导致文字颠倒、比对结果异常。

    Args:
        template_bgr_raw, scan_bgr_raw: 模板和扫描件的原始渲染 BGR。
        rotation:
            - "clockwise": 顺时针 90°（默认）
            - "counterclockwise": 逆时针 90°
            - "auto": 自动检测旋转方向（对横版页做快速配准+war 差，
              选误差小的方向）

    Returns:
        (template_bgr, scan_bgr) 方向归一化后的图；竖版页直接返回原图。
    """
    th, tw = template_bgr_raw.shape[:2]
    sh, sw = scan_bgr_raw.shape[:2]
    template_landscape = tw > th
    scan_landscape = sw > sh

    if not template_landscape and not scan_landscape:
        return template_bgr_raw, scan_bgr_raw

    if rotation == "auto":
        code = _best_landscape_rotation(template_bgr_raw, scan_bgr_raw)
    elif rotation == "counterclockwise":
        code = cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        code = cv2.ROTATE_90_CLOCKWISE

    t_rot = (
        cv2.rotate(template_bgr_raw, code) if template_landscape else template_bgr_raw
    )
    s_rot = cv2.rotate(scan_bgr_raw, code) if scan_landscape else scan_bgr_raw
    return t_rot, s_rot


def _best_landscape_rotation(
    template_bgr_raw: np.ndarray, scan_bgr_raw: np.ndarray
) -> int:
    """AUTO 用：对横版页试两种旋转方向，做快速配准+war 差，选误差小的方向。

    适用于模板和扫描件横放方向可能不同（顺/逆时针 90° 横放）的场景。
    文本方向检测（OCR）成本高且对模糊扫描件不稳，所以用配准反馈判断：
    旋转后配准+war 差小的方向 = 文本对齐正确的方向。

    Returns:
        cv2.ROTATE_90_CLOCKWISE 或 cv2.ROTATE_90_COUNTERCLOCKWISE。
    """
    scale = 0.5
    t_small = cv2.resize(
        template_bgr_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    s_small = cv2.resize(
        scan_bgr_raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )

    tg = cv2.cvtColor(t_small, cv2.COLOR_BGR2GRAY)
    sg = cv2.cvtColor(s_small, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=1500)
    kp_s, d_s = sift.detectAndCompute(sg, None)
    kp_t, d_t = sift.detectAndCompute(tg, None)
    if d_s is None or d_t is None or len(kp_s) < 15 or len(kp_t) < 15:
        return cv2.ROTATE_90_CLOCKWISE

    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 30})
    best_code = cv2.ROTATE_90_CLOCKWISE
    best_diff = float("inf")

    for code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
        t_rot = (
            cv2.rotate(t_small, code)
            if template_bgr_raw.shape[1] > template_bgr_raw.shape[0]
            else t_small
        )
        s_rot = (
            cv2.rotate(s_small, code)
            if scan_bgr_raw.shape[1] > scan_bgr_raw.shape[0]
            else s_small
        )
        tg_r = cv2.cvtColor(t_rot, cv2.COLOR_BGR2GRAY)
        sg_r = cv2.cvtColor(s_rot, cv2.COLOR_BGR2GRAY)
        kp_r, d_r = sift.detectAndCompute(sg_r, None)
        kp_t_r, d_t_r = sift.detectAndCompute(tg_r, None)
        if d_r is None or d_t_r is None:
            continue
        raw_r = matcher.knnMatch(d_r, d_t_r, k=2)
        good_r = [p[0] for p in raw_r if len(p) == 2 and p[0].distance < 0.70 * p[1].distance]
        if len(good_r) < 10:
            continue
        ps = np.float32([kp_r[m.queryIdx].pt for m in good_r]).reshape(-1, 1, 2)
        pt = np.float32([kp_t_r[m.trainIdx].pt for m in good_r]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(ps, pt, cv2.RANSAC, 2.0)
        if H is None:
            continue
        th_r, tw_r = tg_r.shape[:2]
        warped = cv2.warpPerspective(
            sg_r, H, (tw_r, th_r), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        diff = float(np.mean(cv2.absdiff(warped, tg_r)))
        if diff < best_diff:
            best_diff = diff
            best_code = code

    return best_code


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
        patch_output_dir: str | Path | None = None,
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
            patch_output_dir:      候选框 patch 数据集输出目录（可选）

        Returns:
            PixelDiffResult: 包含差异列表、配准指标、耗时的完整结果。
        """
        # ── 阶段1：输入 ──
        timer = StageTimer()
        _ensure_logging(self.config)
        logger.info(
            "compare start scan=%s template=%s page=%d dpi=%d",
            scan_path, template_path, page, self.config.dpi,
        )

        # 渲染文档页为 BGR uint8（300 DPI）
        template_bgr_raw = load_document_page_bgr(template_path, page, self.config.dpi)
        scan_bgr_raw = load_document_page_bgr(scan_path, page, self.config.dpi)
        validate_image_array(template_bgr_raw, "template")
        validate_image_array(scan_bgr_raw, "scan")
        # 方向归一化：横版页统一旋转为竖版（扫描 PDF 常见横放页）。
        # 注意：归一化只用于比对，viewer 展示的原始图（template/candidate）
        # 仍输出归一化前的原始渲染，避免横版页显示方向被旋转。
        template_bgr, scan_bgr = _apply_orientation(
            template_bgr_raw, scan_bgr_raw, self.config.alignment_orientation_rotation
        )

        # 记录模板尺寸，后续所有差异坐标以此为基准
        template_height, template_width = template_bgr.shape[:2]
        self.config.validate_for_image(template_width, template_height)
        unchanged_text_mask = build_unchanged_text_mask(
            scan_path=scan_path,
            template_path=template_path,
            page=page,
            image_shape=(template_height, template_width),
            config=self.config,
        )
        image_keep_mask = build_pdf_image_keep_mask(
            scan_path=scan_path,
            template_path=template_path,
            page=page,
            image_shape=(template_height, template_width),
            config=self.config,
        )
        render_ms = timer.checkpoint("render")
        logger.debug(
            "stage render %dms template=%dx%d unchanged_text_mask=%s image_keep_mask=%s",
            render_ms, template_width, template_height,
            unchanged_text_mask is not None, image_keep_mask is not None,
        )

        # ── 阶段2：预处理 ──
        # 2a. 颜色过滤：去除扫描件上的红色公章和蓝色手写签名
        color_protection_mask = (
            unchanged_text_mask
            if unchanged_text_mask is not None and unchanged_text_mask.shape == scan_bgr.shape[:2]
            else None
        )
        filtered_scan = remove_colored_marks_bgr(
            scan_bgr,
            self.config,
            protected_zero_mask=color_protection_mask,
        )
        color_ms = timer.checkpoint("color_filter")
        logger.debug("stage color_filter %dms", color_ms)

        text_anchor_lines = extract_text_anchor_lines(scan_path, template_path, page, self.config)
        if not text_anchor_lines and not has_pdf_text_layer(scan_path, page, self.config.dpi):
            text_anchor_lines = extract_ocr_text_anchor_lines(
                filtered_scan, template_bgr, str(template_path), page, self.config
            )
        text_anchor_alignment = align_by_text_anchors_bgr(
            filtered_scan, template_bgr, text_anchor_lines, self.config
        )
        filtered_scan = text_anchor_alignment.aligned_bgr
        anchor_ms = timer.checkpoint("text_anchor_alignment")
        logger.debug(
            "stage text_anchor_alignment %dms anchors=%d applied_lines=%d",
            anchor_ms, text_anchor_alignment.anchors, text_anchor_alignment.applied_lines,
        )

        # 2b. SURF/FLANN/RANSAC 全局配准：将扫描件对齐到模板坐标系
        alignment = align_scan_to_template_bgr(filtered_scan, template_bgr, self.config)
        align_ms = timer.checkpoint("global_alignment")
        logger.info(
            "alignment detector=%s fallback=%s good_matches=%d inlier_ratio=%.4f "
            "elapsed_ms=%d downsampled=%s",
            alignment.detector, alignment.detector_fallback, alignment.good_matches,
            alignment.inlier_ratio, align_ms, alignment.feature_downsampled,
        )
        line_alignment = align_text_lines_by_centroid_bgr(
            alignment.aligned_bgr,
            template_bgr,
            self.config,
        )
        affine_alignment = align_text_lines_affine_bgr(
            line_alignment.aligned_bgr,
            template_bgr,
            list(line_alignment.matched_line_centers),
            self.config,
        )
        line_ms = timer.checkpoint("line_alignment")
        logger.debug(
            "stage line_alignment %dms applied=%s matched_pairs=%d",
            line_ms, line_alignment.applied, line_alignment.matched_pairs,
        )
        piecewise_alignment = align_text_lines_piecewise_bgr(
            affine_alignment.aligned_bgr,
            template_bgr,
            list(line_alignment.matched_line_centers),
            self.config,
        )
        piecewise_ms = timer.checkpoint("piecewise_alignment")
        logger.debug(
            "stage piecewise_alignment %dms applied_lines=%d rigid_blocks=%d",
            piecewise_ms, piecewise_alignment.applied_lines,
            piecewise_alignment.rigid_blocks_applied,
        )
        local_warp = apply_constrained_local_warp_bgr(
            piecewise_alignment.aligned_bgr,
            template_bgr,
            self.config,
        )
        aligned_bgr = local_warp.aligned_bgr
        warp_ms = timer.checkpoint("local_warp")
        logger.debug(
            "stage local_warp %dms applied=%s max_disp=%.2f gate_skipped=%s",
            warp_ms, local_warp.applied, local_warp.max_displacement,
            local_warp.gate_skipped,
        )
        aligned_gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

        # 可选：输出调试用中间图像
        metadata: dict[str, Any] = {}
        if template_output_path is not None:
            # 模板展示图输出方向归一化后的图（与待检图方向一致）。
            # 同样用 JPEG+缩放到 1600px（避免 750K~975K PNG 翻页卡），
            # viewer 显示几乎无差，体积降到几百 KB。
            template_out = Path(template_output_path)
            template_out.parent.mkdir(parents=True, exist_ok=True)
            preview = template_bgr
            h, w = template_bgr.shape[:2]
            longest = max(h, w)
            if longest > 1600:
                scale = 1600 / longest
                preview = cv2.resize(
                    template_bgr,
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(
                str(template_out),
                preview,
                [cv2.IMWRITE_JPEG_QUALITY, 82],
            )
            metadata["template_output_path"] = str(template_out.resolve())
        if candidate_output_path is not None:
            # 待检文档输出「配准对齐后」的扫描件（aligned_bgr），与模板/残影图
            # 处于同一模板坐标系、同一尺寸。差异框坐标本就是模板坐标系，只有
            # 用对齐后的图，viewer 悬停高亮才不会因扫描件与模板的几何错位而偏移。
            # 扫描件是灰度照片性质，PNG 无损压不动（实测 2-3MB/页）；
            # 缩到最长边 1600px + JPEG（质量 82），viewer 显示几乎无差，
            # 体积降到几百 KB，翻页不再卡顿。
            candidate_out = Path(candidate_output_path)
            candidate_out.parent.mkdir(parents=True, exist_ok=True)
            preview = aligned_bgr
            h, w = aligned_bgr.shape[:2]
            longest = max(h, w)
            if longest > 1600:
                scale = 1600 / longest
                preview = cv2.resize(
                    aligned_bgr,
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(
                str(candidate_out),
                preview,
                [cv2.IMWRITE_JPEG_QUALITY, 82],
            )
            metadata["candidate_output_path"] = str(candidate_out.resolve())

        # ── 阶段3：二值化 ──
        # 扫描件用自适应阈值（应对光照不均），模板用 Otsu
        scan_binary = binarize_scan_bgr(aligned_bgr, self.config, gray=aligned_gray)
        template_binary = binarize_template_bgr(template_bgr, gray=template_gray)
        binary_ms = timer.checkpoint("binarization")
        logger.debug("stage binarization %dms", binary_ms)

        # ── 阶段4：差异检测 ──
        # 4a. XOR 异或：暴露所有像素级差异
        diff = xor_difference(scan_binary, template_binary)
        residual_alignment = realign_residual_text_lines_bgr(
            aligned_bgr,
            template_bgr,
            scan_binary,
            template_binary,
            diff,
            list(line_alignment.matched_line_centers),
            self.config,
        )
        aligned_bgr = residual_alignment.aligned_bgr
        scan_binary = residual_alignment.scan_binary
        diff = residual_alignment.diff_mask
        if residual_alignment.applied_lines:
            aligned_gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        residual_ms = timer.checkpoint("residual_line_alignment")
        logger.debug(
            "stage residual_line_alignment %dms applied_lines=%d diff_pixels=%d->%d",
            residual_ms, residual_alignment.applied_lines,
            residual_alignment.before_diff_pixels, residual_alignment.after_diff_pixels,
        )
        if image_keep_mask is not None:
            diff = cv2.bitwise_and(diff, image_keep_mask)
        # 4b. 边缘裁剪：抑制扫描仪边框伪影（40px）
        diff = crop_edges(diff, self.config.crop_margin)
        # 4c. 形态学清理：去噪 + 碎片合并
        diff, recall_diff = prepare_difference_masks(
            diff,
            unchanged_text_mask,
            self.config,
        )
        sensitive_recall_regions = extract_sensitive_text_recall_regions(
            template_path=template_path,
            page=page,
            diff_mask=recall_diff,
            config=self.config,
        )
        sensitive_recall_regions = filter_recalled_similarity_regions(
            sensitive_recall_regions,
            scan_binary=scan_binary,
            template_binary=template_binary,
            config=self.config,
        )
        diff_ms = timer.checkpoint("difference_text")
        logger.debug(
            "stage difference_text %dms diff_pixels=%d recall_regions=%d",
            diff_ms, int(cv2.countNonZero(diff)), len(sensitive_recall_regions),
        )

        # ── 阶段5：区域分析 ──
        # 5a. 轮廓提取：从差异掩码提取外轮廓 → DifferenceRegion 列表
        regions = extract_regions(diff, self.config.min_diff_area)
        filter_metrics: dict[str, int | float | str] = {}
        if self.config.multilevel_filter_enabled:
            filter_result = apply_multilevel_filters(
                regions,
                aligned_bgr=aligned_bgr,
                template_bgr=template_bgr,
                scan_binary=scan_binary,
                template_binary=template_binary,
                scan_path=scan_path,
                template_path=template_path,
                page=page,
                config=self.config,
                scan_gray=aligned_gray,
                template_gray=template_gray,
            )
            regions = filter_result.regions
            filter_metrics = filter_result.metrics
            if filter_result.annotations:
                metadata["text_annotations"] = filter_result.annotations
        else:
            # 5b. 局部相似性过滤：7 级策略剔除配准残余假阳性
            regions = filter_locally_similar_regions(
                regions,
                scan_binary=scan_binary,
                template_binary=template_binary,
                config=self.config,
            )
        text_regions = extract_text_difference_regions(
            scan_path=scan_path,
            template_path=template_path,
            page=page,
            image_shape=diff.shape[:2],
            config=self.config,
        )
        if text_regions:
            regions = merge_regions(regions, text_regions, config=self.config)
        similarity_template_text = extract_template_text_by_region(
            regions=regions,
            template_path=template_path,
            page=page,
            dpi=self.config.dpi,
            padding=0,
        )
        regions = filter_recalled_similarity_regions(
            regions,
            scan_binary=scan_binary,
            template_binary=template_binary,
            config=self.config,
            template_text_by_region=similarity_template_text,
        )
        if sensitive_recall_regions:
            regions = merge_regions(regions, sensitive_recall_regions, config=self.config)
        filter_ms = timer.checkpoint("filtering")
        logger.info(
            "filtering regions=%d elapsed_ms=%d metrics=%s",
            len(regions), filter_ms, filter_metrics,
        )

        risk_metrics: dict[str, int] = {}
        if self.config.risk_review_enabled:
            template_text_by_region = extract_template_text_by_region(
                regions=regions,
                template_path=template_path,
                page=page,
                dpi=self.config.dpi,
                padding=self.config.risk_review_region_padding,
            )
            risk_result = apply_risk_review(
                regions=regions,
                aligned_bgr=aligned_bgr,
                template_text_by_region=template_text_by_region,
                config=self.config,
                template_bgr=template_bgr,
                scan_gray=aligned_gray,
                template_gray=template_gray,
            )
            regions = risk_result.regions
            risk_metrics = risk_result.metrics
            # 展示用「原文」精确重提：小外扩（6px，补偿配准错位但不跨行），
            # 只取真正与差异框相交的文本，避免上一行/下一行混入。风险等级已
            # 在上面用宽范围文本判定完成，这里仅覆盖展示字段，不影响 risk_level。
            precise_text = extract_template_text_by_region(
                regions=regions,
                template_path=template_path,
                page=page,
                dpi=self.config.dpi,
                padding=6,
            )
            regions = [
                replace(r, template_text=precise_text.get(r.id))
                for r in regions
            ]
        regions, displacement_pairs = pair_displaced_regions(
            regions,
            scan_binary,
            template_binary,
            self.config,
        )
        regions = classify_difference_regions(
            regions,
            scan_binary,
            template_binary,
            self.config,
        )
        # 移除像素变化量不足的渲染噪声假阳性
        # 双重保险：只有同时满足 (变化小 + 模板无文字 + 非高风险) 才丢弃
        if self.config.min_significant_pixel_change > 0:
            removed = 0
            kept: list[object] = []
            for r in regions:
                total_change = (r.added_pixels or 0) + (r.deleted_pixels or 0)
                has_text = bool(getattr(r, "template_text", None))
                is_high = (getattr(r, "risk_level", None) == "HIGH")
                if total_change >= self.config.min_significant_pixel_change or has_text or is_high:
                    kept.append(r)
                else:
                    removed += 1
            if removed:
                logger.info(
                    "filtered %d insignificant pixel regions (threshold=%d)",
                    removed, self.config.min_significant_pixel_change,
                )
                regions = kept
        risk_ms = timer.checkpoint("risk_review")
        logger.info(
            "risk_review regions=%d displacement_pairs=%d elapsed_ms=%d metrics=%s",
            len(regions), displacement_pairs, risk_ms, risk_metrics,
        )

        # ── 阶段6：可视化输出 ──
        visual_path: str | None = None
        if regions and visual_output_path is not None:
            visual = draw_regions(
                template_bgr,
                regions,
                show_classification_labels=(
                    self.config.difference_classification_enabled
                    and self.config.difference_classification_labels_enabled
                ),
            )
            visual_path = write_image_bgr(visual_output_path, visual)

        if ghost_output_path is not None:
            ghost = draw_text_ghost_comparison(
                template_binary,
                scan_binary,
                match_tolerance=self.config.ghost_match_tolerance,
            )
            if self.config.ghost_region_boxes and regions:
                ghost = draw_regions(
                    ghost,
                    regions,
                    show_classification_labels=(
                        self.config.difference_classification_enabled
                        and self.config.difference_classification_labels_enabled
                    ),
                )
            metadata["ghost_output_path"] = write_image_bgr(ghost_output_path, ghost)
        if patch_output_dir is not None:
            patch_dir = export_region_patches(
                regions=regions,
                template_bgr=template_bgr,
                scan_bgr=aligned_bgr,
                output_dir=patch_output_dir,
                page=page,
                config=self.config,
            )
            if patch_dir is not None:
                metadata["patch_output_dir"] = str(patch_dir)

        # ── 构建结果 ──
        output_ms = timer.checkpoint("output")
        logger.debug("stage output %dms", output_ms)
        timing_metrics = timer.finish()
        elapsed_ms = timing_metrics["elapsed_ms"]
        result = PixelDiffResult(
            status="completed",
            page=page,
            image={"width": template_width, "height": template_height, "dpi": self.config.dpi},
            differences=regions,
            metrics={
                **timing_metrics,
                "good_matches": alignment.good_matches,
                "inlier_ratio": alignment.inlier_ratio,
                "feature_detector": alignment.detector,
                "feature_detector_fallback": int(alignment.detector_fallback),
                "text_anchor_checked_lines": text_anchor_alignment.checked_lines,
                "text_anchor_applied_lines": text_anchor_alignment.applied_lines,
                "text_anchor_count": text_anchor_alignment.anchors,
                "text_anchor_protected_intervals": text_anchor_alignment.protected_intervals,
                "text_anchor_before_iou": text_anchor_alignment.before_iou,
                "text_anchor_after_iou": text_anchor_alignment.after_iou,
                "alignment_feature_downsampled": int(alignment.feature_downsampled),
                "alignment_feature_scale": alignment.feature_scale,
                "alignment_feature_fallback": int(alignment.feature_downsample_fallback),
                "line_centroid_alignment": int(self.config.line_centroid_alignment),
                "line_centroid_applied": int(line_alignment.applied),
                "line_centroid_scan_lines": line_alignment.scan_lines,
                "line_centroid_template_lines": line_alignment.template_lines,
                "line_centroid_matched_pairs": line_alignment.matched_pairs,
                "line_centroid_max_abs_offset": line_alignment.max_abs_offset,
                "line_horizontal_applied": int(line_alignment.horizontal_applied),
                "line_horizontal_anchors": line_alignment.horizontal_anchors,
                "line_horizontal_max_abs_offset": line_alignment.max_abs_horizontal_offset,
                "line_affine_applied_lines": affine_alignment.applied_lines,
                "line_affine_checked_lines": affine_alignment.checked_lines,
                "line_affine_before_iou": affine_alignment.before_iou,
                "line_affine_after_iou": affine_alignment.after_iou,
                "line_affine_max_scale_delta": affine_alignment.max_scale_delta,
                "line_affine_max_displacement": affine_alignment.max_displacement,
                "line_piecewise_applied_lines": piecewise_alignment.applied_lines,
                "line_piecewise_checked_lines": piecewise_alignment.checked_lines,
                "line_piecewise_anchors": piecewise_alignment.anchors,
                "line_piecewise_protected_intervals": piecewise_alignment.protected_intervals,
                "line_piecewise_before_iou": piecewise_alignment.before_iou,
                "line_piecewise_after_iou": piecewise_alignment.after_iou,
                "line_piecewise_max_displacement": piecewise_alignment.max_displacement,
                "line_piecewise_max_scale_delta": piecewise_alignment.max_scale_delta,
                "rigid_text_blocks_attempted": piecewise_alignment.rigid_blocks_attempted,
                "rigid_text_blocks_applied": piecewise_alignment.rigid_blocks_applied,
                "rigid_text_blocks_rejected_overlap": (
                    piecewise_alignment.rigid_blocks_rejected_overlap
                ),
                "rigid_text_blocks_rejected_quality": (
                    piecewise_alignment.rigid_blocks_rejected_quality
                ),
                "rigid_text_block_applied_lines": (
                    piecewise_alignment.rigid_block_applied_lines
                ),
                "residual_line_candidate_lines": residual_alignment.candidate_lines,
                "residual_line_attempted_lines": residual_alignment.attempted_lines,
                "residual_line_applied_lines": residual_alignment.applied_lines,
                "residual_line_before_diff_pixels": residual_alignment.before_diff_pixels,
                "residual_line_after_diff_pixels": residual_alignment.after_diff_pixels,
                "residual_line_before_long_residuals": residual_alignment.before_long_residuals,
                "residual_line_after_long_residuals": residual_alignment.after_long_residuals,
                "residual_line_protected_intervals": residual_alignment.protected_intervals,
                "residual_line_protected_retention": residual_alignment.protected_retention,
                "residual_line_max_displacement": residual_alignment.max_displacement,
                "residual_line_max_scale_delta": residual_alignment.max_scale_delta,
                "local_warp_enabled": int(self.config.local_warp_enabled),
                "local_warp_applied": int(local_warp.applied),
                "local_warp_max_displacement": local_warp.max_displacement,
                "local_warp_mean_displacement": local_warp.mean_displacement,
                "local_warp_gate_skipped": int(local_warp.gate_skipped),
                "local_warp_gate_foreground_iou": local_warp.gate_foreground_iou,
                "displacement_pairs": displacement_pairs,
                **filter_metrics,
                **risk_metrics,
            },
            visual_output_path=visual_path,
            metadata=metadata,
        )

        # 可选：输出 JSON 结果
        if json_output_path is not None:
            write_json(json_output_path, result.to_dict())

        logger.info(
            "compare completed page=%d regions=%d elapsed_ms=%d good_matches=%d "
            "inlier_ratio=%.4f",
            page, len(regions), elapsed_ms, alignment.good_matches,
            alignment.inlier_ratio,
        )
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
    patch_output_dir: str | Path | None = None,
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
        patch_output_dir=patch_output_dir,
    )
