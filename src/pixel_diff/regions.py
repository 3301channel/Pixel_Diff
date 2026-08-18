"""连通域提取与局部相似性多级过滤。

区域分析阶段包含两个部分：

1. extract_regions()           — 从差异掩码提取外轮廓，生成 DifferenceRegion 列表
2. filter_locally_similar_regions() — 7 级局部相似性过滤，剔除配准残余假阳性

核心设计思想：
  如果对差异区域的局部子图做微小平移后，模板和扫描件能高度重合，
  说明差异来自配准残余而非真实内容修改 —— 应当丢弃。
"""

from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.models import DifferenceRegion, PixelDiffConfig
from pixel_diff.region_utils import renumber_regions


def extract_regions(diff_mask: np.ndarray, min_area: float) -> list[DifferenceRegion]:
    """从差异掩码中提取外轮廓，生成带编号的差异区域列表。

    处理流程：
    1. cv2.findContours(RETR_EXTERNAL) — 只取最外层轮廓
    2. 过滤面积 < min_area 的轮廓
    3. 计算每个轮廓的外接矩形，裁剪到图像边界内
    4. 按 (y, x, -area) 排序（自上而下、自左而右、面积大的在前）
    5. 从 1 开始编号
    """
    height, width = diff_mask.shape[:2]

    # 提取外轮廓（RETR_EXTERNAL 不取嵌套内轮廓，CHAIN_APPROX_SIMPLE 压缩水平/垂直/对角线段）
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        # 外接矩形并裁剪到图像边界
        x, y, box_width, box_height = cv2.boundingRect(contour)
        x = max(0, min(int(x), width - 1))
        y = max(0, min(int(y), height - 1))
        box_width = max(1, min(int(box_width), width - x))
        box_height = max(1, min(int(box_height), height - y))
        raw.append((x, y, box_width, box_height, area))

    # 排序：主键 y（从上到下），次键 x（从左到右），第三键 -area（面积大的在前）
    raw.sort(key=lambda item: (item[1], item[0], -item[4]))

    return [
        DifferenceRegion(
            id=index,
            x=x,
            y=y,
            width=box_width,
            height=box_height,
            area=area,
        )
        for index, (x, y, box_width, box_height, area) in enumerate(raw, start=1)
    ]


def filter_locally_similar_regions(
    regions: list[DifferenceRegion],
    scan_binary: np.ndarray,
    template_binary: np.ndarray,
    config: PixelDiffConfig,
) -> list[DifferenceRegion]:
    """7 级局部相似性过滤：逐级剔除配准残余导致的假阳性差异。

    对每个差异区域依次执行：

    第1级  _is_long_horizontal_residual  — 长水平残差（宽高比≥12，高度≤20px）
        无需计算 IoU，仅凭形状即可判决。

    计算   _local_foreground_density       — 局部前景密度（padding=40px 窗口）

    第2级  _is_sparse_residual            — 稀疏/小残差
        - 面积 200~400px² 且密度 ≤ 0.04
        - 面积 200~220px² 且密度 ≤ 0.12

    计算   _best_local_iou                 — 局部 IoU 平移搜索（±4px，81 次）
        探询："轻微平移后，两份图是否高度重合？"
        （search_radius=4，共 81 次尝试）

    第3级  _is_short_horizontal_residual  — 短水平残差（宽高比≥2.5，高≤20，IoU≥0.45）
    第4级  _is_wide_text_residual         — 宽文本残差（面积≥5000，宽高比≥3，IoU≥0.20）
    第5级  IoU ≥ 0.55                      — 通用高 IoU 过滤

    全部未命中 → 保留为疑似真实差异。
    """
    # 过滤器关闭或阈值为 0 时直接返回
    if (
        not config.local_similarity_filter
        or config.local_similarity_iou_threshold <= 0
        or not regions
    ):
        return renumber_regions(regions)

    if scan_binary.shape != template_binary.shape:
        return renumber_regions(regions)

    # ── 反转前景/背景：0(文字)→1，255(背景)→0 ──
    # 后续 IOu 和密度计算在"有字=1"的前景图上进行
    scan_foreground = (scan_binary == 0).astype(np.uint8)
    template_foreground = (template_binary == 0).astype(np.uint8)

    kept = []
    for region in regions:
        # 第1级：长水平残差（仅形状判断，无计算开销）
        if _is_long_horizontal_residual(region, config):
            continue

        # 计算：局部前景密度（需 padding=40px 上下文窗口）
        density = _local_foreground_density(
            region,
            scan_foreground,
            template_foreground,
            padding=config.residual_density_padding,
        )

        # 第2级：稀疏/小残差
        if _is_sparse_residual(region, density, config):
            continue

        # 计算：局部 IoU 平移搜索（核心计算，约 81 次 warpAffine）
        iou_search = (
            _best_local_iou_vectorized
            if config.local_similarity_vectorized_search_enabled
            else _best_local_iou
        )
        local_iou = iou_search(
            region,
            scan_foreground,
            template_foreground,
            padding=config.local_similarity_padding,
            search_radius=config.local_similarity_search_radius,
        )

        # 第3级：短水平残差
        if _is_short_horizontal_residual(region, local_iou, config):
            continue

        # 第4级：整行宽文本残差。只对宽行候选做更大半径搜索，避免误删小改动。
        if _is_wide_row_residual_candidate(region, config):
            wide_row_iou = iou_search(
                region,
                scan_foreground,
                template_foreground,
                padding=config.local_similarity_padding,
                search_radius=config.wide_row_residual_search_radius,
            )
            if wide_row_iou >= config.wide_row_residual_min_iou:
                continue

        # 第4级：宽文本残差
        if _is_wide_text_residual(region, local_iou, config):
            continue

        # 第5级：通用高 IoU
        if local_iou >= config.local_similarity_iou_threshold:
            continue

        # 全部未命中 → 保留
        kept.append(region)

    return renumber_regions(kept)


def _best_local_iou(
    region: DifferenceRegion,
    scan_foreground: np.ndarray,
    template_foreground: np.ndarray,
    padding: int,
    search_radius: int,
) -> float:
    """对差异区域局部子图做平移搜索，返回最佳 IoU。

    工作方式：
    1. 以区域为中心，向外扩展 padding 像素，裁剪局部子图
    2. 对模板子图做 dx∈[-r,+r], dy∈[-r,+r] 的平移（共 (2r+1)² 次）
    3. 每次平移后计算 scan 和 shifted_template 的 IoU
    4. 返回所有平移中的最大值

    IoU = 交集面积 / 并集面积

    如最佳 IoU 接近 1.0 → 平移后高度重合 → 配准残余
    如最佳 IoU 很低     → 平移也无法对上 → 可能是真实修改
    """
    height, width = scan_foreground.shape[:2]

    # 裁剪区域（padding 扩展，限制在图像边界内）
    x0 = max(0, region.x - padding)
    y0 = max(0, region.y - padding)
    x1 = min(width, region.x + region.width + padding)
    y1 = min(height, region.y + region.height + padding)

    scan_crop = scan_foreground[y0:y1, x0:x1]
    template_crop = template_foreground[y0:y1, x0:x1]

    if scan_crop.size == 0 or template_crop.size == 0:
        return 1.0  # 空区域直接返回 1.0（视为一致）

    crop_height, crop_width = scan_crop.shape[:2]
    best_iou = 0.0

    # 极速 IoU 计算：bool 位运算 + count_nonzero 替代 logical_and/or + sum。
    # 实测比 logical_and(...).sum() 快约 2.5 倍（避免中间临时数组 + 命中 bool 快速路径）。
    scan_mask = scan_crop.astype(bool)
    template_mask = template_crop.astype(bool)

    # 遍历所有平移组合
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            # 2×3 仿射变换矩阵（纯平移）
            transform = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

            # 用 INTER_NEAREST 保持二值语义（不产生浮点插值）
            shifted_template = cv2.warpAffine(
                template_mask.astype(np.uint8),
                transform,
                (crop_width, crop_height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,  # 平移后露出的边界填 0（无文字）
            ).astype(bool)

            # 位运算统计像素：交集/并集
            intersection = np.count_nonzero(scan_mask & shifted_template)
            union = np.count_nonzero(scan_mask | shifted_template)

            iou = 1.0 if union == 0 else float(intersection / union)
            if iou > best_iou:
                best_iou = iou

    return best_iou


def _best_local_iou_vectorized(
    region: DifferenceRegion,
    scan_foreground: np.ndarray,
    template_foreground: np.ndarray,
    padding: int,
    search_radius: int,
) -> float:
    """Return the legacy local-IoU maximum using one native correlation call."""
    height, width = scan_foreground.shape[:2]
    x0 = max(0, region.x - padding)
    y0 = max(0, region.y - padding)
    x1 = min(width, region.x + region.width + padding)
    y1 = min(height, region.y + region.height + padding)
    scan_crop = scan_foreground[y0:y1, x0:x1]
    template_crop = template_foreground[y0:y1, x0:x1]
    if scan_crop.size == 0 or template_crop.size == 0:
        return 1.0

    radius = search_radius
    padded = cv2.copyMakeBorder(
        template_crop,
        radius,
        radius,
        radius,
        radius,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    intersections = cv2.matchTemplate(
        padded.astype(np.float32),
        scan_crop.astype(np.float32),
        cv2.TM_CCORR,
    )
    intersections = np.rint(intersections).astype(np.int64)

    crop_height, crop_width = scan_crop.shape
    integral = cv2.integral(padded, sdepth=cv2.CV_32S)
    template_sums = (
        integral[crop_height:, crop_width:]
        - integral[:-crop_height, crop_width:]
        - integral[crop_height:, :-crop_width]
        + integral[:-crop_height, :-crop_width]
    ).astype(np.int64)
    scan_sum = int(scan_crop.sum())
    unions = scan_sum + template_sums - intersections
    scores = np.divide(
        intersections,
        unions,
        out=np.ones_like(intersections, dtype=np.float64),
        where=unions != 0,
    )
    return float(scores.max())


def _is_long_horizontal_residual(region: DifferenceRegion, config: PixelDiffConfig) -> bool:
    """第1级：长水平残差过滤器。

    条件：宽高比 ≥ 12 且 高度 ≤ 20px
    物理含义：一整行文字由于配准偏移产生的水平细长条纹。
    仅凭形状即可判定，无需计算 IoU。
    """
    if region.height <= 0:
        return False
    aspect = region.width / region.height
    return (
        config.horizontal_residual_min_aspect > 0
        and aspect >= config.horizontal_residual_min_aspect
        and region.height <= config.horizontal_residual_max_height
    )


def _is_short_horizontal_residual(
    region: DifferenceRegion,
    local_iou: float,
    config: PixelDiffConfig,
) -> bool:
    """第3级：短水平残差过滤器。

    条件：宽高比 ≥ 2.5 且 高度 ≤ 20px 且 IoU ≥ 0.45
    捕的是较短的（不一定整行的）水平偏移残余。
    """
    if region.height <= 0:
        return False
    aspect = region.width / region.height
    return (
        config.short_horizontal_residual_min_aspect > 0
        and aspect >= config.short_horizontal_residual_min_aspect
        and region.height <= config.short_horizontal_residual_max_height
        and local_iou >= config.short_horizontal_residual_min_iou
    )


def _is_wide_text_residual(
    region: DifferenceRegion,
    local_iou: float,
    config: PixelDiffConfig,
) -> bool:
    """第4级：宽文本残差过滤器。

    条件：面积 ≥ 5000px² 且 宽高比 ≥ 3.0 且 IoU ≥ 0.20
    捕的是大块横排文字区域的整体偏移。
    IoU 阈值较低（0.30），因为大区域内自由空间多、文字密度低。
    """
    if region.height <= 0:
        return False
    aspect = region.width / region.height
    return (
        config.wide_text_residual_min_area > 0
        and region.area >= config.wide_text_residual_min_area
        and aspect >= config.wide_text_residual_min_aspect
        and local_iou >= config.wide_text_residual_min_iou
    )


def _is_wide_row_residual_candidate(
    region: DifferenceRegion,
    config: PixelDiffConfig,
) -> bool:
    """Return true for full-line residual candidates that merit a larger search."""
    if region.height <= 0:
        return False
    return (
        config.wide_row_residual_min_width > 0
        and config.wide_row_residual_search_radius > config.local_similarity_search_radius
        and region.width >= config.wide_row_residual_min_width
        and region.height <= config.wide_row_residual_max_height
        and region.area >= config.wide_text_residual_min_area
    )


def _is_sparse_residual(
    region: DifferenceRegion,
    local_density: float,
    config: PixelDiffConfig,
) -> bool:
    """第2级：稀疏/小残差过滤器。

    两个子检查：

    A（稀疏残差）：面积 200~400px² 且 局部前景密度 ≤ 0.04
      区域不大不小，但周围几乎没有文字 —— 典型的孤立的配准噪声

    B（小残差）：面积 200~220px² 且 局部前景密度 ≤ 0.12
      区域偏小，周围文字密度也不高
    """
    # 检查 A：稀疏残差
    if (
        config.sparse_residual_max_area > 0
        and region.area >= config.residual_filter_min_area
        and region.area <= config.sparse_residual_max_area
        and local_density <= config.sparse_residual_max_density
    ):
        return True

    # 检查 B：小残差
    return (
        config.small_residual_max_area > 0
        and region.area >= config.residual_filter_min_area
        and region.area <= config.small_residual_max_area
        and local_density <= config.small_residual_max_density
    )


def _local_foreground_density(
    region: DifferenceRegion,
    scan_foreground: np.ndarray,
    template_foreground: np.ndarray,
    padding: int,
) -> float:
    """计算差异区域周围的局部前景密度。

    以区域为中心，向外扩展 padding 像素，统计扩展后的窗口内
    扫描件和模板的前景（文字）像素占比。

    密度 = (扫描件前景像素数 + 模板前景像素数) / 窗口总面积

    高密度 → 周围有很多文字 → 可能是真实的文本修改
    低密度 → 周围几乎没文字  → 差异区域孤零零的，可能是噪声
    """
    height, width = scan_foreground.shape[:2]

    # 扩展后的窗口坐标
    x0 = max(0, region.x - padding)
    y0 = max(0, region.y - padding)
    x1 = min(width, region.x + region.width + padding)
    y1 = min(height, region.y + region.height + padding)

    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        return 0.0

    # 统计窗口内前景像素数
    scan_count = int(scan_foreground[y0:y1, x0:x1].sum())
    template_count = int(template_foreground[y0:y1, x0:x1].sum())

    return float((scan_count + template_count) / area)
