"""类型化数据模型与配置验证。

定义：
- PixelDiffConfig   — 算法全量参数，支持 YAML 加载和运行时校验
- DifferenceRegion  — 单个疑似差异区域（id + 外接矩形 + 面积）
- PixelDiffResult   — 单页比对结果（状态 + 差异列表 + 诊断指标）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import yaml

from pixel_diff.exceptions import ConfigurationError

# 类型别名
KernelSize: TypeAlias = tuple[int, int]
"""形态学核尺寸 (宽, 高)。"""

HSVRange: TypeAlias = tuple[tuple[int, int, int], tuple[int, int, int]]
"""HSV 范围 ((h_low, s_low, v_low), (h_high, s_high, v_high))。"""


@dataclass(frozen=True)
class PixelDiffConfig:
    """算法全量可配置参数，使用 frozen dataclass 确保不可变性。

    所有参数均有默认值（300 DPI 下的经验调优值），
    支持通过 YAML 文件覆盖部分参数。
    """

    # ─── 渲染 ───
    dpi: int = 300
    """PDF 渲染精度（Dots Per Inch），影响输出图像分辨率。"""

    # ─── 日志 ───
    log_level: str = "INFO"
    """控制台日志级别：DEBUG / INFO / WARNING / ERROR / CRITICAL。"""

    log_console: bool = True
    """是否将日志输出到控制台（stderr）。"""

    log_file: bool = True
    """是否生成日志文件（文件级别固定为 DEBUG，最详细）。"""

    # ─── 颜色过滤（去除红章蓝签）───
    filter_colored_marks: bool = True
    """是否启用 HSV 色彩过滤去除红色公章和蓝色手写签名。"""

    red_hsv_ranges: tuple[HSVRange, ...] = (
        ((0, 40, 40), (10, 255, 255)),
        ((170, 40, 40), (180, 255, 255)),
    )
    """红色 HSV 范围：色环两端，H∈[0,10]∪[170,180]，S≥40，V≥40。"""

    blue_hsv_ranges: tuple[HSVRange, ...] = (((100, 40, 40), (124, 255, 255)),)
    """蓝色 HSV 范围：H∈[100,124]，S≥40，V≥40。"""

    # ─── 特征配准 ───
    feature_detector: str = "surf"
    """首选特征检测器：surf 或 sift。"""

    feature_detector_fallback: str = "sift"
    """首选检测器不可用时的回退检测器；留空表示不回退。"""

    surf_hessian_threshold: float = 400.0
    """SURF Hessian 阈值。值越小提取的特征点越多。"""

    sift_nfeatures: int = 10000
    """SIFT 最大特征点数量。回退到 SIFT 时使用。"""

    lowe_ratio: float = 0.70
    """Lowe 比率测试阈值：d1/d2 < 0.70 才保留匹配。"""

    min_good_matches: int = 15
    """通过 Lowe 测试的最少匹配点数，不足则抛出 AlignmentError。"""

    ransac_reprojection_threshold: float = 3.0
    """RANSAC 重投影误差阈值（像素），判定内点/外点的距离上限。"""

    alignment_feature_downsample_enabled: bool = False
    """Estimate feature homography on reduced images before one full-resolution warp."""

    alignment_feature_scale: float = 0.5
    """Feature-image scale in (0, 1]."""

    alignment_feature_fallback_enabled: bool = True
    """Retry feature alignment at full resolution when the reduced attempt is weak."""

    alignment_feature_min_inlier_ratio: float = 0.8
    """Minimum reduced-attempt RANSAC inlier ratio before accepting its homography."""

    alignment_orientation_rotation: str = "clockwise"
    """横版页旋转方向：'clockwise' 顺时针 / 'counterclockwise' 逆时针 / 'auto' 自动。
    不同扫描件的横放方向可能不同（顺时针 90° 横放 vs 逆时针 90° 横放），
    错误的旋转会导致文字颠倒。auto 用配准+差异判断旋转方向。"""

    blank_page_alignment_enabled: bool = False
    """Use identity alignment when either page has effectively no ink."""

    blank_page_ink_threshold: int = 245
    blank_page_max_ink_ratio: float = 0.00001

    # ─── 差异检测 ───
    crop_margin: int = 40
    """边缘裁剪宽度（像素），抑制扫描仪边框伪影。"""

    min_diff_area: float = 200.0
    """差异区域最小面积（像素），面积不足的不计入结果。"""

    min_significant_pixel_change: int = 100
    """区域中增加+删除像素的最小合计值，低于此视为渲染噪声并丢弃。

    两份内容相同但渲染引擎不同的 PDF，会产生大量 <100 像素的
    抗锯齿/字体微调假阳性。设为 0 恢复全部检出。
    """

    difference_classification_enabled: bool = True
    difference_direction_ratio_threshold: float = 0.80
    difference_classification_labels_enabled: bool = True
    large_modified_as_displaced_enabled: bool = True
    large_modified_min_page_area_ratio: float = 0.003
    large_modified_min_aspect_ratio: float = 6.0
    large_modified_min_direction_balance: float = 0.60
    """差异区域最小面积（像素），面积不足的不计入结果。"""

    # ─── 文本行质心补偿 ───
    line_centroid_alignment: bool = False
    """是否启用文本行质心纵向补偿。默认关闭，便于试验和撤销。"""

    line_centroid_max_drift: int = 30
    """文本行质心匹配允许的最大垂直漂移像素。"""

    line_centroid_row_dilate_width: int = 180
    """文本行提取时的横向膨胀核宽度。"""

    line_centroid_row_dilate_height: int = 4
    """文本行提取时的纵向膨胀核高度。"""

    line_centroid_min_width_ratio: float = 0.25
    """文本行连通域最小宽度占页面宽度比例。"""

    line_centroid_min_height: int = 8
    """文本行连通域最小高度。"""

    line_centroid_max_height: int = 100
    """文本行连通域最大高度。"""

    line_centroid_consistency_filter: bool = False
    """是否启用行质心匹配一致性过滤。默认关闭，便于试验和撤销。"""

    line_centroid_median_tolerance: int = 10
    """行质心匹配偏移相对全局中位数允许的最大偏差像素。"""

    line_horizontal_alignment: bool = False
    """是否启用置信度约束的文本行横向平移校正。"""

    line_horizontal_max_shift: int = 12
    """单条文本行允许的最大水平校正像素。"""

    line_horizontal_band_half_height: int = 36
    """估计水平偏移时文本行上下各截取的像素数。"""

    line_horizontal_min_iou: float = 0.45
    """接受文本行水平偏移所需的最低二值前景 IoU。"""

    line_horizontal_min_improvement: float = 0.05
    """最佳水平偏移相对零偏移所需的最低 IoU 提升。"""

    line_affine_alignment_enabled: bool = False
    line_affine_window_width: int = 320
    line_affine_window_step: int = 180
    line_affine_max_shift: int = 40
    line_affine_min_anchors: int = 3
    line_affine_min_anchor_iou: float = 0.20
    line_affine_min_improvement: float = 0.02
    line_affine_max_scale_delta: float = 0.04
    line_affine_band_half_height: int = 32

    line_piecewise_alignment_enabled: bool = False
    line_piecewise_window_width: int = 96
    line_piecewise_window_step: int = 48
    line_piecewise_max_shift: int = 30
    line_piecewise_min_anchor_similarity: float = 0.55
    line_piecewise_min_anchors: int = 4
    line_piecewise_jump_threshold: int = 8
    line_piecewise_max_scale_delta: float = 0.15
    line_piecewise_protection_width: int = 24
    line_piecewise_min_improvement: float = 0.02

    rigid_text_block_alignment_enabled: bool = False
    rigid_text_block_min_gap_width: int = 4
    rigid_text_block_max_internal_gap: int = 10
    rigid_text_block_min_anchor_similarity: float = 0.60
    rigid_text_block_min_iou_improvement: float = 0.01

    residual_line_realignment_enabled: bool = False
    residual_line_window_width: int = 64
    residual_line_window_step: int = 24
    residual_line_max_shift: int = 24
    residual_line_min_anchor_similarity: float = 0.50
    residual_line_min_anchors: int = 4
    residual_line_min_span: int = 240
    residual_line_min_components: int = 4
    residual_line_jump_threshold: int = 7
    residual_line_protection_width: int = 24
    residual_line_max_scale_delta: float = 0.18
    residual_line_min_iou_improvement: float = 0.01
    residual_line_min_diff_reduction: float = 0.08
    residual_line_min_protected_retention: float = 0.80

    # ─── 风险复核 ───
    local_warp_enabled: bool = False
    """Enable experimental constrained dense local displacement compensation."""

    local_warp_max_displacement: float = 6.0
    """Maximum local warp displacement in pixels."""

    local_warp_scale: float = 0.35
    """Downscale factor for dense flow estimation, in (0, 1]."""

    local_warp_blur_kernel: int = 31
    """Odd Gaussian smoothing kernel for local warp flow; 0 disables smoothing."""

    local_warp_gate_enabled: bool = False
    """Skip dense flow when a cheap foreground comparison is already highly confident."""

    local_warp_gate_min_iou: float = 0.985
    """Foreground IoU at or above which dense local flow is skipped."""

    ssim_early_exit_enabled: bool = False
    """Stop translation search once the consumer's filtering threshold is reached."""

    report_parallel_workers: int = 1
    """Number of report pages compared in separate worker processes."""

    ghost_match_tolerance: int = 0
    """Visualization-only stroke matching tolerance in pixels."""

    ghost_region_boxes: bool = False
    """Draw numbered detected-region boxes on the ghost image."""

    risk_review_enabled: bool = False
    """是否启用差异框文本风险复核。默认关闭，便于试验和撤销。"""

    risk_review_filter_low: bool = False
    """风险复核后是否过滤 LOW 风险区域。默认关闭，避免误删新增内容。"""

    risk_review_ocr_enabled: bool = False
    """是否启用可选 OCR 注释。OCR 不可用时自动跳过。"""

    risk_review_region_padding: int = 8
    """提取模板文本和 OCR patch 时对差异框外扩的像素。"""

    risk_review_ocr_padding: int = 40
    """OCR patch extraction padding. Keep this larger than text-region padding."""

    risk_review_ocr_match_filter_enabled: bool = True
    """Downgrade non-sensitive text residuals when OCR confirms template text."""

    risk_review_ocr_match_ratio: float = 0.9
    """Minimum normalized text similarity for OCR/template confirmation."""

    risk_review_plain_text_residual_enabled: bool = True
    """是否将结构相似的普通正文小残留降级为 LOW。"""

    risk_review_plain_text_max_area: float = 3500.0
    """普通正文结构残留降级的最大区域面积。"""

    risk_review_plain_text_ssim_threshold: float = 0.72
    """普通正文结构残留降级所需的最低局部 SSIM。"""

    risk_review_ssim_padding: int = 15
    """风险复核局部 SSIM 的区域外扩像素。"""

    risk_review_ssim_search_radius: int = 4
    """风险复核局部 SSIM 平移搜索半径。"""

    risk_review_plain_text_protected_chars: str = "甲申大犬太夫天未末土士日曰目田由"
    """普通正文结构残留中禁止降级的近形汉字集合。"""

    risk_review_stroke_match_enabled: bool = False
    """Downgrade bounded same-shape text residuals using stroke distance."""

    risk_review_stroke_match_tolerance: int = 4
    risk_review_stroke_match_min_coverage: float = 0.94
    risk_review_stroke_match_min_area: float = 3000.0
    risk_review_stroke_match_max_area: float = 6500.0
    risk_review_stroke_match_padding: int = 12

    risk_review_narrow_stroke_enabled: bool = False
    """Downgrade tightly bounded high-coverage text-stroke registration residuals."""

    risk_review_narrow_stroke_max_area: float = 500.0
    risk_review_narrow_stroke_max_width: int = 20
    risk_review_narrow_stroke_max_height: int = 32
    risk_review_narrow_stroke_min_coverage: float = 0.96

    risk_review_page_number_match_enabled: bool = False
    """Downgrade unchanged numeric markers in the bottom-center page-number zone."""

    risk_review_page_number_bottom_ratio: float = 0.88
    risk_review_page_number_center_tolerance_ratio: float = 0.12
    risk_review_page_number_max_width: int = 120
    risk_review_page_number_max_height: int = 100
    risk_review_page_number_shape_tolerance: int = 3
    risk_review_page_number_min_coverage: float = 0.95

    # ─── 神经网络辅助数据导出 ───
    risk_review_large_visual_residual_enabled: bool = True
    """Preserve large visual-only residuals when text/OCR signals are unavailable."""

    risk_review_large_visual_min_area: float = 2500.0
    """Minimum contour area for preserving a visual-only residual."""

    risk_review_large_visual_min_width: int = 60
    """Minimum box width for preserving a visual-only residual."""

    risk_review_large_visual_min_height: int = 30
    """Minimum box height for preserving a visual-only residual."""

    risk_review_watermark_filter_enabled: bool = True
    """Keep large low-contrast background watermark residuals eligible for LOW filtering."""

    risk_review_watermark_max_p95_delta: float = 100.0
    """Maximum 95th percentile grayscale delta for a residual to look like a watermark."""

    risk_review_watermark_max_very_dark_ratio: float = 0.05
    """Maximum very-dark pixel ratio in the scan crop for watermark classification."""

    risk_review_watermark_max_template_dark_ratio: float = 0.08
    """Maximum dark pixel ratio in the template crop for watermark classification."""

    risk_review_watermark_dark_threshold: int = 140
    """Scan grayscale threshold used for very-dark watermark protection."""

    risk_review_watermark_template_dark_threshold: int = 210
    """Template grayscale threshold used to ensure the residual is on background."""

    # ─── 未重叠字符形过滤：只保留字符形，横线/竖线/噪点降级为 LOW ───
    risk_review_char_shape_filter_enabled: bool = True
    """对 visual_difference_without_text 区域做字符形过滤（横线/竖线/噪点降级）。"""

    risk_review_char_min_aspect_ratio: float = 0.2
    """宽高比下限（width/height），低于此值视为竖线。"""

    risk_review_char_max_aspect_ratio: float = 5.0
    """宽高比上限（width/height），高于此值视为横线。"""

    risk_review_char_min_area: float = 100.0
    """包围盒面积下限，低于此值视为噪点。"""

    patch_export_enabled: bool = False
    """是否导出候选框 patch 数据，用于人工标注和神经网络训练。"""

    patch_export_padding: int = 24
    """导出 patch 时对差异框外扩的像素。"""

    # ─── 局部相似性过滤 ───
    # PDF text-layer assist
    pdf_text_layer_filter: bool = True
    """Use extractable PDF characters to suppress unchanged text residuals."""

    pdf_text_position_tolerance: int = 4
    """Max pixel bbox delta for equal PDF characters to be treated as stable."""

    pdf_text_mask_padding: int = 5
    """Padding around stable PDF character boxes when masking pixel residuals."""

    pdf_text_region_padding: int = 8
    """Padding around supplemental PDF text-difference boxes."""

    merge_region_expand_padding: int = 0
    """Max px gap allowed between a supplemental text region and a pixel region
    for them to be merged into one box. 0 = only merge on real (overlapping)
    intersection. Higher values absorb appended/adjacent characters into the
    parent word's box (causing them to disappear as separate regions)."""

    pdf_image_region_filter_enabled: bool = False
    """Ignore embedded image rectangles on text-bearing PDF pages."""

    pdf_image_region_padding: int = 3
    pdf_image_region_max_page_ratio: float = 0.80

    pdf_text_anchor_alignment_enabled: bool = False
    pdf_text_anchor_min_equal_chars: int = 3
    pdf_text_anchor_protection_padding: int = 10
    pdf_text_anchor_max_shift: int = 80
    pdf_text_anchor_max_scale_delta: float = 0.08
    pdf_text_anchor_min_improvement: float = 0.02
    pdf_text_anchor_ocr_fallback_enabled: bool = False
    pdf_text_anchor_ocr_scale: float = 0.5
    pdf_text_anchor_ocr_min_confidence: float = 0.90
    pdf_text_anchor_ocr_max_before_iou: float = 0.10
    pdf_text_anchor_ocr_max_page_iou: float = 0.035

    displacement_pairing_enabled: bool = False
    """Merge matching added/deleted residuals into one displacement region."""

    displacement_pairing_min_direction_ratio: float = 0.80
    displacement_pairing_min_similarity: float = 0.82
    displacement_pairing_max_size_ratio: float = 1.80
    displacement_pairing_padding: int = 3

    sensitive_text_recall_enabled: bool = False
    """Recover fragmented differences inside template digit/Latin text runs."""

    sensitive_text_recall_padding: int = 8
    """Padding around a sensitive template text run."""

    sensitive_text_recall_min_density: float = 0.02
    """Minimum nonzero difference ratio required inside a sensitive text ROI."""

    sensitive_recall_similarity_filter_enabled: bool = False
    """Recheck recalled regions with a bounded binary foreground IoU search."""

    sensitive_recall_similarity_search_radius: int = 16
    """Maximum translation radius for ordinary recalled text regions."""

    sensitive_recall_page_number_search_radius: int = 64
    """Maximum translation radius for bottom-centered numeric page numbers."""

    sensitive_recall_similarity_iou_threshold: float = 0.62
    """Drop a recalled region when its best translated foreground IoU reaches this value."""

    multilevel_filter_enabled: bool = False
    """是否启用试验性三层过滤。默认关闭，避免 SSIM 带来性能下降。"""

    text_annotation_enabled: bool = True
    """三层过滤开启时，是否为区域生成 PDF text-layer 注释。"""

    colored_residual_filter_enabled: bool = True
    """三层过滤 Level 1：是否过滤红/蓝彩色文字或标注的残差。"""

    colored_residual_min_ratio: float = 0.20
    """区域局部红/蓝像素占比达到该值时视为彩色残差。"""

    colored_residual_padding: int = 8
    """彩色残差判断时的区域外扩像素。"""

    isolated_residual_filter_enabled: bool = False
    """三层过滤 Level 1：是否过滤远离其它差异的孤立小残差。"""

    isolated_residual_max_area: float = 1300.0
    """孤立小残差最大面积。"""

    isolated_residual_max_width: int = 45
    """孤立小残差最大宽度。"""

    isolated_residual_max_height: int = 45
    """孤立小残差最大高度。"""

    isolated_residual_min_neighbor_distance: float = 90.0
    """小残差与最近其它差异中心距离大于该值时视为孤立。"""

    isolated_residual_max_density: float = 0.45
    """孤立小残差局部前景密度上限，避免删除实心新增块。"""

    same_line_merge_filter_enabled: bool = True
    """三层过滤 Level 1：是否将同行邻近小残差并入较大残差。"""

    same_line_merge_small_area: float = 800.0
    """同行合并中可被并入的小残差最大面积。"""

    same_line_merge_max_gap: int = 36
    """同行合并中两个框的最大水平间隙。"""

    same_line_merge_max_center_y_delta: int = 18
    """同行合并中两个框中心 Y 的最大差值。"""

    local_similarity_filter: bool = True
    """是否启用局部相似性多级过滤（主开关）。"""

    local_similarity_iou_threshold: float = 0.62
    """通用 IoU 阈值：局部最佳 IoU ≥ 0.62 的视为配准残余，过滤为背景。"""

    local_similarity_padding: int = 8
    """局部 IoU 搜索时向外扩展的 padding 像素数。"""

    local_similarity_search_radius: int = 4
    local_similarity_vectorized_search_enabled: bool = False
    """Compute all local-IoU translations with correlation and integral sums."""
    """局部 IoU 平移搜索半径（±4px，共 81 次尝试）。"""

    # ─── 局部相似性：水平残差过滤 ───
    horizontal_residual_min_aspect: float = 12.0
    """长水平残差过滤：最小宽高比。"""

    horizontal_residual_max_height: int = 20
    """长水平残差过滤：最大高度（像素）。"""

    short_horizontal_residual_min_aspect: float = 2.5
    """短水平残差过滤：最小宽高比。"""

    short_horizontal_residual_max_height: int = 20
    """短水平残差过滤：最大高度（像素）。"""

    short_horizontal_residual_min_iou: float = 0.55
    """短水平残差过滤：局部 IoU 最低阈值。"""

    wide_text_residual_min_area: float = 5000.0
    """宽文本残差过滤：最小面积（像素）。"""

    wide_text_residual_min_aspect: float = 3.0
    """宽文本残差过滤：最小宽高比。"""

    wide_text_residual_min_iou: float = 0.30
    """宽文本残差过滤：局部 IoU 最低阈值。"""

    wide_row_residual_min_width: int = 500
    """宽行残差二次过滤：最小宽度（像素）。"""

    wide_row_residual_max_height: int = 90
    """宽行残差二次过滤：最大高度（像素）。"""

    wide_row_residual_search_radius: int = 16
    """宽行残差二次过滤：更大的局部平移搜索半径。"""

    wide_row_residual_min_iou: float = 0.45
    """宽行残差二次过滤：大半径搜索后的最低 IoU。"""

    # ─── 局部相似性：稀疏残差过滤 ───
    sparse_residual_max_area: float = 400.0
    """稀疏残差过滤 A：最大面积。"""

    sparse_residual_max_density: float = 0.04
    """稀疏残差过滤 A：最高局部前景密度。"""

    small_residual_max_area: float = 220.0
    """稀疏残差过滤 B（小残差）：最大面积。"""

    small_residual_max_density: float = 0.12
    """稀疏残差过滤 B（小残差）：最高局部前景密度。"""

    residual_filter_min_area: float = 200.0
    """稀疏残差过滤的通用最小面积门槛。"""

    residual_density_padding: int = 40
    """局部前景密度计算时的外扩 padding 像素数。"""

    ssim_filter_enabled: bool = True
    """三层过滤开启时，是否启用 Level 2 局部灰度 SSIM 结构过滤。"""

    ssim_filter_threshold: float = 0.96
    """SSIM 分数达到该阈值时视为结构一致的配准残差。"""

    ssim_filter_padding: int = 15
    """SSIM 局部 patch 外扩像素。"""

    ssim_filter_search_radius: int = 4
    """SSIM 局部平移搜索半径。"""

    ssim_filter_min_region_area: float = 400.0
    ssim_cached_template_stats_enabled: bool = False
    """Reuse template SSIM moments across translation candidates."""
    """低于该面积的区域跳过 SSIM，避免弱信号误判。"""

    # ─── 二值化 ───
    adaptive_block_size: int = 21
    """自适应阈值邻域大小（须为 >1 的奇数）。"""

    adaptive_c: int = 5
    """自适应阈值偏移量 C，从加权均值中减去的常数。"""

    median_blur_kernel: int = 3
    """中值滤波核大小（须为奇数，0=跳过）。"""

    bilateral_diameter: int = 9
    """双边滤波邻域直径。"""

    bilateral_sigma_color: float = 75.0
    """双边滤波灰度域标准差。越大则更多灰度差异被平滑。"""

    bilateral_sigma_space: float = 75.0
    """双边滤波空间域标准差。越大则更远像素参与平滑。"""

    # ─── 形态学 ───
    min_noise_component_area: float = 12.0
    """小连通域删除阈值（像素面积），低于此值的组件视为噪声。"""

    open_kernel: KernelSize = (3, 3)
    close_kernel: KernelSize = (3, 3)
    dilate_kernel: KernelSize = (15, 10)
    """膨胀核尺寸（宽,高），横向 > 纵向适配横排中文。
    较大的核将碎片合并为块，便于后续过滤器命中。"""

    morph_iterations_open: int = 1
    morph_iterations_close: int = 1
    morph_iterations_dilate: int = 1
    """形态学操作的迭代次数。"""

    @classmethod
    def from_yaml(cls, path: str | Path) -> PixelDiffConfig:
        """从 YAML 文件加载配置并校验。

        自动将列表类型的核尺寸转为 tuple。
        """
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        # YAML 加载时 kernel 是列表，需要转 tuple 以匹配 KernelSize
        for key in ("open_kernel", "close_kernel", "dilate_kernel"):
            if key in raw:
                raw[key] = tuple(raw[key])

        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        """校验所有不依赖具体图像尺寸的参数。

        Raises:
            ConfigurationError: 任何参数值不合法时抛出。
        """
        if self.dpi <= 0:
            raise ConfigurationError("configuration: dpi must be positive")
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigurationError(
                "configuration: log_level must be one of DEBUG/INFO/WARNING/ERROR/CRITICAL"
            )
        if not 0 < self.lowe_ratio < 1:
            raise ConfigurationError("configuration: lowe_ratio must be in (0, 1)")
        if self.feature_detector not in {"surf", "sift"}:
            raise ConfigurationError("configuration: feature_detector must be 'surf' or 'sift'")
        if self.feature_detector_fallback not in {"", "surf", "sift"}:
            raise ConfigurationError(
                "configuration: feature_detector_fallback must be '', 'surf', or 'sift'"
            )
        if self.surf_hessian_threshold <= 0:
            raise ConfigurationError("configuration: surf_hessian_threshold must be positive")
        if self.sift_nfeatures <= 0:
            raise ConfigurationError("configuration: sift_nfeatures must be positive")
        if self.min_good_matches < 4:
            raise ConfigurationError("configuration: min_good_matches must be at least 4")
        if self.ransac_reprojection_threshold <= 0:
            raise ConfigurationError(
                "configuration: ransac_reprojection_threshold must be positive"
            )
        if not 0 < self.alignment_feature_scale <= 1:
            raise ConfigurationError("configuration: alignment_feature_scale must be in (0, 1]")
        if not 0 <= self.alignment_feature_min_inlier_ratio <= 1:
            raise ConfigurationError(
                "configuration: alignment_feature_min_inlier_ratio must be in [0, 1]"
            )
        if not 0 <= self.blank_page_ink_threshold <= 255:
            raise ConfigurationError("configuration: blank_page_ink_threshold must be in [0, 255]")
        if not 0 <= self.blank_page_max_ink_ratio <= 1:
            raise ConfigurationError("configuration: blank_page_max_ink_ratio must be in [0, 1]")
        if self.crop_margin < 0:
            raise ConfigurationError("configuration: crop_margin must be non-negative")
        if self.min_diff_area < 0:
            raise ConfigurationError("configuration: min_diff_area must be non-negative")
        if not 0.5 <= self.difference_direction_ratio_threshold <= 1.0:
            raise ConfigurationError(
                "configuration: difference_direction_ratio_threshold must be in [0.5, 1]"
            )
        if not 0 < self.large_modified_min_page_area_ratio <= 1:
            raise ConfigurationError(
                "configuration: large_modified_min_page_area_ratio must be in (0, 1]"
            )
        if self.large_modified_min_aspect_ratio < 1:
            raise ConfigurationError(
                "configuration: large_modified_min_aspect_ratio must be at least 1"
            )
        if not 0 <= self.large_modified_min_direction_balance <= 1:
            raise ConfigurationError(
                "configuration: large_modified_min_direction_balance must be in [0, 1]"
            )
        if self.line_centroid_max_drift < 0:
            raise ConfigurationError("configuration: line_centroid_max_drift must be non-negative")
        if self.line_centroid_row_dilate_width <= 0:
            raise ConfigurationError(
                "configuration: line_centroid_row_dilate_width must be positive"
            )
        if self.line_centroid_row_dilate_height <= 0:
            raise ConfigurationError(
                "configuration: line_centroid_row_dilate_height must be positive"
            )
        if not 0 <= self.line_centroid_min_width_ratio <= 1:
            raise ConfigurationError(
                "configuration: line_centroid_min_width_ratio must be in [0, 1]"
            )
        if self.line_centroid_min_height < 0:
            raise ConfigurationError("configuration: line_centroid_min_height must be non-negative")
        if self.line_centroid_max_height <= self.line_centroid_min_height:
            raise ConfigurationError(
                "configuration: line_centroid_max_height must be greater than "
                "line_centroid_min_height"
            )
        if self.line_centroid_median_tolerance < 0:
            raise ConfigurationError(
                "configuration: line_centroid_median_tolerance must be non-negative"
            )
        if self.line_horizontal_max_shift < 0:
            raise ConfigurationError(
                "configuration: line_horizontal_max_shift must be non-negative"
            )
        if self.line_horizontal_band_half_height <= 0:
            raise ConfigurationError(
                "configuration: line_horizontal_band_half_height must be positive"
            )
        if not 0 <= self.line_horizontal_min_iou <= 1:
            raise ConfigurationError("configuration: line_horizontal_min_iou must be in [0, 1]")
        if not 0 <= self.line_horizontal_min_improvement <= 1:
            raise ConfigurationError(
                "configuration: line_horizontal_min_improvement must be in [0, 1]"
            )
        if (
            min(
                self.line_affine_window_width,
                self.line_affine_window_step,
                self.line_affine_max_shift,
                self.line_affine_band_half_height,
            )
            <= 0
        ):
            raise ConfigurationError("configuration: line affine sizes and shifts must be positive")
        if self.line_affine_min_anchors < 2:
            raise ConfigurationError("configuration: line_affine_min_anchors must be at least 2")
        if not 0 <= self.line_affine_min_anchor_iou <= 1:
            raise ConfigurationError("configuration: line_affine_min_anchor_iou must be in [0, 1]")
        if not 0 <= self.line_affine_min_improvement <= 1:
            raise ConfigurationError("configuration: line_affine_min_improvement must be in [0, 1]")
        if not 0 < self.line_affine_max_scale_delta < 0.5:
            raise ConfigurationError(
                "configuration: line_affine_max_scale_delta must be in (0, 0.5)"
            )
        if min(self.line_piecewise_window_width, self.line_piecewise_window_step) <= 0:
            raise ConfigurationError("configuration: piecewise window sizes must be positive")
        if self.line_piecewise_max_shift < 0 or self.line_piecewise_jump_threshold < 0:
            raise ConfigurationError("configuration: piecewise shifts must be non-negative")
        if not 0 <= self.line_piecewise_min_anchor_similarity <= 1:
            raise ConfigurationError(
                "configuration: line_piecewise_min_anchor_similarity must be in [0, 1]"
            )
        if self.line_piecewise_min_anchors < 2:
            raise ConfigurationError("configuration: line_piecewise_min_anchors must be at least 2")
        if not 0 < self.line_piecewise_max_scale_delta < 0.5:
            raise ConfigurationError(
                "configuration: line_piecewise_max_scale_delta must be in (0, 0.5)"
            )
        if self.line_piecewise_protection_width < 0:
            raise ConfigurationError(
                "configuration: line_piecewise_protection_width must be non-negative"
            )
        if not 0 <= self.line_piecewise_min_improvement <= 1:
            raise ConfigurationError(
                "configuration: line_piecewise_min_improvement must be in [0, 1]"
            )
        if self.rigid_text_block_min_gap_width < 1:
            raise ConfigurationError(
                "configuration: rigid_text_block_min_gap_width must be at least 1"
            )
        if self.rigid_text_block_max_internal_gap < 0:
            raise ConfigurationError(
                "configuration: rigid_text_block_max_internal_gap must be non-negative"
            )
        if not 0 <= self.rigid_text_block_min_anchor_similarity <= 1:
            raise ConfigurationError(
                "configuration: rigid_text_block_min_anchor_similarity must be in [0, 1]"
            )
        if not 0 <= self.rigid_text_block_min_iou_improvement <= 1:
            raise ConfigurationError(
                "configuration: rigid_text_block_min_iou_improvement must be in [0, 1]"
            )
        if min(self.residual_line_window_width, self.residual_line_window_step) <= 0:
            raise ConfigurationError("configuration: residual line window sizes must be positive")
        if min(
            self.residual_line_max_shift,
            self.residual_line_min_span,
            self.residual_line_jump_threshold,
            self.residual_line_protection_width,
        ) < 0:
            raise ConfigurationError("configuration: residual line sizes must be non-negative")
        if self.residual_line_min_anchors < 2 or self.residual_line_min_components < 2:
            raise ConfigurationError(
                "configuration: residual line anchor/component counts must be at least 2"
            )
        if not 0 <= self.residual_line_min_anchor_similarity <= 1:
            raise ConfigurationError(
                "configuration: residual_line_min_anchor_similarity must be in [0, 1]"
            )
        if not 0 < self.residual_line_max_scale_delta < 0.5:
            raise ConfigurationError(
                "configuration: residual_line_max_scale_delta must be in (0, 0.5)"
            )
        for name, value in (
            ("residual_line_min_iou_improvement", self.residual_line_min_iou_improvement),
            ("residual_line_min_diff_reduction", self.residual_line_min_diff_reduction),
            ("residual_line_min_protected_retention", self.residual_line_min_protected_retention),
        ):
            if not 0 <= value <= 1:
                raise ConfigurationError(f"configuration: {name} must be in [0, 1]")
        if self.risk_review_region_padding < 0:
            raise ConfigurationError(
                "configuration: risk_review_region_padding must be non-negative"
            )
        if self.risk_review_ocr_padding < 0:
            raise ConfigurationError("configuration: risk_review_ocr_padding must be non-negative")
        if not 0 <= self.risk_review_ocr_match_ratio <= 1:
            raise ConfigurationError("configuration: risk_review_ocr_match_ratio must be in [0, 1]")
        if self.risk_review_plain_text_max_area < 0:
            raise ConfigurationError(
                "configuration: risk_review_plain_text_max_area must be non-negative"
            )
        if not 0 <= self.risk_review_plain_text_ssim_threshold <= 1:
            raise ConfigurationError(
                "configuration: risk_review_plain_text_ssim_threshold must be in [0, 1]"
            )
        if self.risk_review_ssim_padding < 0:
            raise ConfigurationError("configuration: risk_review_ssim_padding must be non-negative")
        if self.risk_review_ssim_search_radius < 0:
            raise ConfigurationError(
                "configuration: risk_review_ssim_search_radius must be non-negative"
            )
        if self.risk_review_stroke_match_tolerance < 0:
            raise ConfigurationError(
                "configuration: risk_review_stroke_match_tolerance must be non-negative"
            )
        if not 0 <= self.risk_review_stroke_match_min_coverage <= 1:
            raise ConfigurationError(
                "configuration: risk_review_stroke_match_min_coverage must be in [0, 1]"
            )
        if self.risk_review_stroke_match_min_area < 0:
            raise ConfigurationError(
                "configuration: risk_review_stroke_match_min_area must be non-negative"
            )
        if self.risk_review_stroke_match_max_area < self.risk_review_stroke_match_min_area:
            raise ConfigurationError(
                "configuration: stroke-match maximum area must be at least minimum area"
            )
        if self.risk_review_stroke_match_padding < 0:
            raise ConfigurationError(
                "configuration: risk_review_stroke_match_padding must be non-negative"
            )
        if self.risk_review_narrow_stroke_max_area < 0:
            raise ConfigurationError(
                "configuration: risk_review_narrow_stroke_max_area must be non-negative"
            )
        if self.risk_review_narrow_stroke_max_width < 0:
            raise ConfigurationError(
                "configuration: risk_review_narrow_stroke_max_width must be non-negative"
            )
        if self.risk_review_narrow_stroke_max_height < 0:
            raise ConfigurationError(
                "configuration: risk_review_narrow_stroke_max_height must be non-negative"
            )
        if not 0 <= self.risk_review_narrow_stroke_min_coverage <= 1:
            raise ConfigurationError(
                "configuration: risk_review_narrow_stroke_min_coverage must be in [0, 1]"
            )
        if not 0 < self.risk_review_page_number_bottom_ratio < 1:
            raise ConfigurationError(
                "configuration: risk_review_page_number_bottom_ratio must be in (0, 1)"
            )
        if not 0 <= self.risk_review_page_number_center_tolerance_ratio <= 0.5:
            raise ConfigurationError(
                "configuration: page-number center tolerance must be in [0, 0.5]"
            )
        if (
            min(
                self.risk_review_page_number_max_width,
                self.risk_review_page_number_max_height,
            )
            <= 0
        ):
            raise ConfigurationError(
                "configuration: page-number maximum dimensions must be positive"
            )
        if self.risk_review_page_number_shape_tolerance < 0:
            raise ConfigurationError(
                "configuration: page-number shape tolerance must be non-negative"
            )
        if not 0 <= self.risk_review_page_number_min_coverage <= 1:
            raise ConfigurationError(
                "configuration: page-number minimum coverage must be in [0, 1]"
            )
        if self.patch_export_padding < 0:
            raise ConfigurationError("configuration: patch_export_padding must be non-negative")
        if self.risk_review_large_visual_min_area < 0:
            raise ConfigurationError(
                "configuration: risk_review_large_visual_min_area must be non-negative"
            )
        if self.risk_review_large_visual_min_width < 0:
            raise ConfigurationError(
                "configuration: risk_review_large_visual_min_width must be non-negative"
            )
        if self.risk_review_large_visual_min_height < 0:
            raise ConfigurationError(
                "configuration: risk_review_large_visual_min_height must be non-negative"
            )
        if self.risk_review_watermark_max_p95_delta < 0:
            raise ConfigurationError(
                "configuration: risk_review_watermark_max_p95_delta must be non-negative"
            )
        if not 0 <= self.risk_review_watermark_max_very_dark_ratio <= 1:
            raise ConfigurationError(
                "configuration: risk_review_watermark_max_very_dark_ratio must be in [0, 1]"
            )
        if not 0 <= self.risk_review_watermark_max_template_dark_ratio <= 1:
            raise ConfigurationError(
                "configuration: risk_review_watermark_max_template_dark_ratio must be in [0, 1]"
            )
        if not 0 <= self.risk_review_watermark_dark_threshold <= 255:
            raise ConfigurationError(
                "configuration: risk_review_watermark_dark_threshold must be in [0, 255]"
            )
        if not 0 <= self.risk_review_watermark_template_dark_threshold <= 255:
            raise ConfigurationError(
                "configuration: risk_review_watermark_template_dark_threshold must be in [0, 255]"
            )
        if self.local_warp_max_displacement < 0:
            raise ConfigurationError(
                "configuration: local_warp_max_displacement must be non-negative"
            )
        if not 0 < self.local_warp_scale <= 1:
            raise ConfigurationError("configuration: local_warp_scale must be in (0, 1]")
        if self.local_warp_blur_kernel != 0 and (
            self.local_warp_blur_kernel <= 1 or self.local_warp_blur_kernel % 2 == 0
        ):
            raise ConfigurationError(
                "configuration: local_warp_blur_kernel must be 0 or an odd integer greater than 1"
            )
        if not 0 <= self.local_warp_gate_min_iou <= 1:
            raise ConfigurationError("configuration: local_warp_gate_min_iou must be in [0, 1]")
        if self.report_parallel_workers < 1:
            raise ConfigurationError("configuration: report_parallel_workers must be at least 1")
        if self.ghost_match_tolerance < 0:
            raise ConfigurationError("configuration: ghost_match_tolerance must be non-negative")
        if self.pdf_text_position_tolerance < 0:
            raise ConfigurationError(
                "configuration: pdf_text_position_tolerance must be non-negative"
            )
        if self.pdf_text_mask_padding < 0:
            raise ConfigurationError("configuration: pdf_text_mask_padding must be non-negative")
        if self.pdf_text_region_padding < 0:
            raise ConfigurationError("configuration: pdf_text_region_padding must be non-negative")
        if self.merge_region_expand_padding < 0:
            raise ConfigurationError(
                "configuration: merge_region_expand_padding must be non-negative")
        if self.pdf_image_region_padding < 0:
            raise ConfigurationError("configuration: pdf_image_region_padding must be non-negative")
        if not 0 < self.pdf_image_region_max_page_ratio <= 1:
            raise ConfigurationError(
                "configuration: pdf_image_region_max_page_ratio must be in (0, 1]"
            )
        if self.pdf_text_anchor_min_equal_chars < 2:
            raise ConfigurationError(
                "configuration: pdf_text_anchor_min_equal_chars must be at least 2"
            )
        if min(self.pdf_text_anchor_protection_padding, self.pdf_text_anchor_max_shift) < 0:
            raise ConfigurationError(
                "configuration: PDF text anchor distances must be non-negative"
            )
        if not 0 < self.pdf_text_anchor_max_scale_delta < 0.5:
            raise ConfigurationError(
                "configuration: pdf_text_anchor_max_scale_delta must be in (0, 0.5)"
            )
        if not 0 <= self.pdf_text_anchor_min_improvement <= 1:
            raise ConfigurationError(
                "configuration: pdf_text_anchor_min_improvement must be in [0, 1]"
            )
        if not 0 < self.pdf_text_anchor_ocr_scale <= 1:
            raise ConfigurationError("configuration: pdf_text_anchor_ocr_scale must be in (0, 1]")
        if not 0 <= self.pdf_text_anchor_ocr_min_confidence <= 1:
            raise ConfigurationError(
                "configuration: pdf_text_anchor_ocr_min_confidence must be in [0, 1]"
            )
        if not 0 <= self.pdf_text_anchor_ocr_max_before_iou <= 1:
            raise ConfigurationError(
                "configuration: pdf_text_anchor_ocr_max_before_iou must be in [0, 1]"
            )
        if not 0 <= self.pdf_text_anchor_ocr_max_page_iou <= 1:
            raise ConfigurationError(
                "configuration: pdf_text_anchor_ocr_max_page_iou must be in [0, 1]"
            )
        if not 0.5 <= self.displacement_pairing_min_direction_ratio <= 1:
            raise ConfigurationError(
                "configuration: displacement_pairing_min_direction_ratio must be in [0.5, 1]"
            )
        if not 0 <= self.displacement_pairing_min_similarity <= 1:
            raise ConfigurationError(
                "configuration: displacement_pairing_min_similarity must be in [0, 1]"
            )
        if self.displacement_pairing_max_size_ratio < 1:
            raise ConfigurationError(
                "configuration: displacement_pairing_max_size_ratio must be at least 1"
            )
        if self.displacement_pairing_padding < 0:
            raise ConfigurationError(
                "configuration: displacement_pairing_padding must be non-negative"
            )
        if self.sensitive_text_recall_padding < 0:
            raise ConfigurationError(
                "configuration: sensitive_text_recall_padding must be non-negative"
            )
        if not 0 <= self.sensitive_text_recall_min_density <= 1:
            raise ConfigurationError(
                "configuration: sensitive_text_recall_min_density must be in [0, 1]"
            )
        if self.sensitive_recall_similarity_search_radius < 0:
            raise ConfigurationError(
                "configuration: sensitive_recall_similarity_search_radius must be non-negative"
            )
        if self.sensitive_recall_page_number_search_radius < 0:
            raise ConfigurationError(
                "configuration: sensitive_recall_page_number_search_radius must be non-negative"
            )
        if not 0 <= self.sensitive_recall_similarity_iou_threshold <= 1:
            raise ConfigurationError(
                "configuration: sensitive_recall_similarity_iou_threshold must be in [0, 1]"
            )
        if not 0 <= self.colored_residual_min_ratio <= 1:
            raise ConfigurationError("configuration: colored_residual_min_ratio must be in [0, 1]")
        if self.colored_residual_padding < 0:
            raise ConfigurationError("configuration: colored_residual_padding must be non-negative")
        if self.isolated_residual_max_area < 0:
            raise ConfigurationError(
                "configuration: isolated_residual_max_area must be non-negative"
            )
        if self.isolated_residual_max_width < 0:
            raise ConfigurationError(
                "configuration: isolated_residual_max_width must be non-negative"
            )
        if self.isolated_residual_max_height < 0:
            raise ConfigurationError(
                "configuration: isolated_residual_max_height must be non-negative"
            )
        if self.isolated_residual_min_neighbor_distance < 0:
            raise ConfigurationError(
                "configuration: isolated_residual_min_neighbor_distance must be non-negative"
            )
        if self.isolated_residual_max_density < 0:
            raise ConfigurationError(
                "configuration: isolated_residual_max_density must be non-negative"
            )
        if self.same_line_merge_small_area < 0:
            raise ConfigurationError(
                "configuration: same_line_merge_small_area must be non-negative"
            )
        if self.same_line_merge_max_gap < 0:
            raise ConfigurationError("configuration: same_line_merge_max_gap must be non-negative")
        if self.same_line_merge_max_center_y_delta < 0:
            raise ConfigurationError(
                "configuration: same_line_merge_max_center_y_delta must be non-negative"
            )
        if not 0 <= self.local_similarity_iou_threshold <= 1:
            raise ConfigurationError(
                "configuration: local_similarity_iou_threshold must be in [0, 1]"
            )
        if self.local_similarity_padding < 0:
            raise ConfigurationError("configuration: local_similarity_padding must be non-negative")
        if self.local_similarity_search_radius < 0:
            raise ConfigurationError(
                "configuration: local_similarity_search_radius must be non-negative"
            )
        if self.horizontal_residual_min_aspect < 0:
            raise ConfigurationError(
                "configuration: horizontal_residual_min_aspect must be non-negative"
            )
        if self.horizontal_residual_max_height < 0:
            raise ConfigurationError(
                "configuration: horizontal_residual_max_height must be non-negative"
            )
        if self.short_horizontal_residual_min_aspect < 0:
            raise ConfigurationError(
                "configuration: short_horizontal_residual_min_aspect must be non-negative"
            )
        if self.short_horizontal_residual_max_height < 0:
            raise ConfigurationError(
                "configuration: short_horizontal_residual_max_height must be non-negative"
            )
        if not 0 <= self.short_horizontal_residual_min_iou <= 1:
            raise ConfigurationError(
                "configuration: short_horizontal_residual_min_iou must be in [0, 1]"
            )
        if self.wide_text_residual_min_area < 0:
            raise ConfigurationError(
                "configuration: wide_text_residual_min_area must be non-negative"
            )
        if self.wide_text_residual_min_aspect < 0:
            raise ConfigurationError(
                "configuration: wide_text_residual_min_aspect must be non-negative"
            )
        if not 0 <= self.wide_text_residual_min_iou <= 1:
            raise ConfigurationError("configuration: wide_text_residual_min_iou must be in [0, 1]")
        if self.wide_row_residual_min_width < 0:
            raise ConfigurationError(
                "configuration: wide_row_residual_min_width must be non-negative"
            )
        if self.wide_row_residual_max_height < 0:
            raise ConfigurationError(
                "configuration: wide_row_residual_max_height must be non-negative"
            )
        if self.wide_row_residual_search_radius < 0:
            raise ConfigurationError(
                "configuration: wide_row_residual_search_radius must be non-negative"
            )
        if not 0 <= self.wide_row_residual_min_iou <= 1:
            raise ConfigurationError("configuration: wide_row_residual_min_iou must be in [0, 1]")
        if self.sparse_residual_max_area < 0:
            raise ConfigurationError("configuration: sparse_residual_max_area must be non-negative")
        if self.sparse_residual_max_density < 0:
            raise ConfigurationError(
                "configuration: sparse_residual_max_density must be non-negative"
            )
        if self.small_residual_max_area < 0:
            raise ConfigurationError("configuration: small_residual_max_area must be non-negative")
        if self.small_residual_max_density < 0:
            raise ConfigurationError(
                "configuration: small_residual_max_density must be non-negative"
            )
        if self.residual_filter_min_area < 0:
            raise ConfigurationError("configuration: residual_filter_min_area must be non-negative")
        if self.residual_density_padding < 0:
            raise ConfigurationError("configuration: residual_density_padding must be non-negative")
        if not 0 <= self.ssim_filter_threshold <= 1:
            raise ConfigurationError("configuration: ssim_filter_threshold must be in [0, 1]")
        if self.ssim_filter_padding < 0:
            raise ConfigurationError("configuration: ssim_filter_padding must be non-negative")
        if self.ssim_filter_search_radius < 0:
            raise ConfigurationError(
                "configuration: ssim_filter_search_radius must be non-negative"
            )
        if self.ssim_filter_min_region_area < 0:
            raise ConfigurationError(
                "configuration: ssim_filter_min_region_area must be non-negative"
            )
        if self.adaptive_block_size <= 1 or self.adaptive_block_size % 2 == 0:
            raise ConfigurationError(
                "configuration: adaptive_block_size must be an odd integer greater than 1"
            )
        if self.median_blur_kernel != 0 and (
            self.median_blur_kernel <= 1 or self.median_blur_kernel % 2 == 0
        ):
            raise ConfigurationError(
                "configuration: median_blur_kernel must be 0 or an odd integer greater than 1"
            )
        if self.min_noise_component_area < 0:
            raise ConfigurationError("configuration: min_noise_component_area must be non-negative")
        for name, kernel in (
            ("open_kernel", self.open_kernel),
            ("close_kernel", self.close_kernel),
            ("dilate_kernel", self.dilate_kernel),
        ):
            _validate_kernel(name, kernel)
        for name, value in (
            ("morph_iterations_open", self.morph_iterations_open),
            ("morph_iterations_close", self.morph_iterations_close),
            ("morph_iterations_dilate", self.morph_iterations_dilate),
        ):
            if value < 0:
                raise ConfigurationError(f"configuration: {name} must be non-negative")

    def validate_for_image(self, width: int, height: int) -> None:
        """校验依赖图像尺寸的参数（如 crop_margin 不能超过半幅图像）。

        应在加载图像后、开始比对前调用。
        """
        self.validate()
        if width <= 0 or height <= 0:
            raise ConfigurationError("configuration: image dimensions must be positive")
        if self.crop_margin >= width / 2 or self.crop_margin >= height / 2:
            raise ConfigurationError(
                "configuration: crop_margin must be smaller than half of image width and height"
            )


@dataclass(frozen=True)
class DifferenceRegion:
    """单个疑似像素级差异区域（模板页面坐标系）。

    坐标 (x, y) 为外接矩形左上角，width/height 为矩形宽高，
    均以模板页面像素为单位。
    """

    id: int
    """差异区域序号（从 1 开始，自上而下、自左而右排列）。"""

    x: int
    y: int
    """外接矩形左上角坐标（模板页面坐标系，像素）。"""

    width: int
    height: int
    """外接矩形宽高（像素）。"""

    area: float
    """差异区域的实际轮廓面积（像素），非外接矩形面积。"""

    risk_level: str | None = None
    """风险等级：HIGH / MEDIUM / LOW / UNKNOWN。"""

    risk_reason: str | None = None
    """风险判定原因。"""

    template_text: str | None = None
    """差异框覆盖到的模板 PDF 文本。"""

    ocr_text: str | None = None
    """可选 OCR 识别文本。"""

    sensitive_type: str | None = None
    """命中的敏感文本类型。"""

    kept: bool | None = None
    """风险复核后是否保留。"""

    change_type: str | None = None
    change_label: str | None = None
    added_pixels: int | None = None
    deleted_pixels: int | None = None
    classification_confidence: float | None = None
    classification_reason: str | None = None

    text_layer_protected: bool = False
    """True if this region originates from PDF text-layer difference extraction.

    Such regions represent confirmed text changes and should not be downgraded
    to LOW or filtered out by risk-review residual detectors.
    """

    def to_dict(self) -> dict[str, int | float | str | bool]:
        """转为 JSON 可序列化的字典。"""
        payload: dict[str, int | float | str | bool] = {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "area": self.area,
        }
        for key, value in (
            ("risk_level", self.risk_level),
            ("risk_reason", self.risk_reason),
            ("template_text", self.template_text),
            ("ocr_text", self.ocr_text),
            ("sensitive_type", self.sensitive_type),
            ("kept", self.kept),
            ("change_type", self.change_type),
            ("change_label", self.change_label),
            ("added_pixels", self.added_pixels),
            ("deleted_pixels", self.deleted_pixels),
            ("classification_confidence", self.classification_confidence),
            ("classification_reason", self.classification_reason),
        ):
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class PixelDiffResult:
    """单页比对完整结果。"""

    status: str
    """比对状态："completed" 或 "partial"。"""

    page: int
    """页面索引（0-based）。"""

    image: dict[str, int]
    """页面图像元信息 {"width": int, "height": int, "dpi": int}。"""

    differences: list[DifferenceRegion]
    """疑似差异区域列表（已排序、编号）。"""

    metrics: dict[str, int | float | str]
    """诊断指标 {"elapsed_ms", "good_matches", "inlier_ratio", "feature_detector"}。"""

    visual_output_path: str | None = None
    """红框标注图输出路径（可选）。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """附加元数据（如 ghost 图路径、template 图路径等）。"""

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 可序列化的字典。"""
        return {
            "status": self.status,
            "page": self.page,
            "image": self.image,
            "differences": [region.to_dict() for region in self.differences],
            "metrics": self.metrics,
            "visual_output_path": self.visual_output_path,
            "metadata": self.metadata,
        }


def _validate_kernel(name: str, kernel: KernelSize) -> None:
    """校验形态学核尺寸：(宽, 高) 均为正整数。"""
    if len(kernel) != 2:
        raise ConfigurationError(f"configuration: {name} must contain width and height")
    width, height = kernel
    if width <= 0 or height <= 0:
        raise ConfigurationError(f"configuration: {name} dimensions must be positive")
