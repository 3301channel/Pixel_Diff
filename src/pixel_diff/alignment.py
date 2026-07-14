"""SIFT/FLANN/RANSAC 全局单应性配准。

将扫描件对齐到模板坐标系，消除由扫描仪引起的平移、旋转、缩放和透视扭曲。

核心流程：
1. SIFT 特征提取  — 扫描件和模板各提取最多 10000 个尺度不变特征点
2. FLANN 匹配     — KD-Tree 近似最近邻搜索，k=2
3. Lowe 比率测试  — 保留 d1/d2 < 0.70 的可靠匹配，剔除模棱两可的配对
4. RANSAC 单应性  — 从含有错误匹配的集合中鲁棒估计 3×3 透视变换矩阵
5. warpPerspective — 将扫描件变换到模板尺寸，边界填白
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np

from pixel_diff.exceptions import AlignmentError
from pixel_diff.models import PixelDiffConfig


@dataclass(frozen=True)
class AlignmentResult:
    """配准结果：对齐后的扫描件 BGR 图像及诊断指标。"""

    aligned_bgr: np.ndarray
    """对齐后的扫描件图像，尺寸与模板一致。"""

    good_matches: int
    """经过 Lowe 比率测试保留的优质匹配点数量。"""

    inlier_ratio: float
    """RANSAC 内点比例 = inlier 数 / good_matches 数，越接近 1 越好。"""

    homography: np.ndarray
    """3×3 透视变换矩阵（单应性），将扫描件坐标映射到模板坐标。"""


def align_scan_to_template_bgr(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    config: PixelDiffConfig,
) -> AlignmentResult:
    """使用全局单应性将扫描件 BGR 图像对齐到模板 BGR 图像。

    整个流程对灰度图操作（SIFT 不依赖颜色），最后对原彩色图做 warp。

    Raises:
        AlignmentError: 特征点为空、优质匹配不足或单应性计算失败时抛出。
    """
    # ── 转灰度图（SIFT 在灰度上运行）──
    scan_gray = cv2.cvtColor(scan_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # ── 1) SIFT 特征提取 ──
    sift_create = cast(Any, cv2.SIFT_create)  # type: ignore[attr-defined]
    sift = sift_create(nfeatures=config.sift_nfeatures)  # 最多提取 nfeatures 个点
    scan_keypoints, scan_descriptors = sift.detectAndCompute(scan_gray, None)
    template_keypoints, template_descriptors = sift.detectAndCompute(template_gray, None)

    # 描述子为空 = 图像几乎没有可检测的结构特征
    if scan_descriptors is None or template_descriptors is None:
        raise AlignmentError("alignment: SIFT descriptors are empty")
    if (
        len(scan_keypoints) < config.min_good_matches
        or len(template_keypoints) < config.min_good_matches
    ):
        raise AlignmentError("alignment: not enough SIFT keypoints")

    # ── 2) FLANN 特征匹配（KD-Tree，k=2 近邻）──
    index_params: dict[str, bool | int | float | str] = {"algorithm": 1, "trees": 5}
    search_params: dict[str, bool | int | float | str] = {"checks": 50}
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = matcher.knnMatch(scan_descriptors, template_descriptors, k=2)

    # ── 3) Lowe 比率测试 ──
    # 原理：最佳匹配的距离应显著小于次佳（d1/d2 < 0.70），
    #       否则说明该点存在多个相似候选，匹配不可靠。
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) != 2:
            continue
        first, second = match_pair
        if first.distance < config.lowe_ratio * second.distance:
            good_matches.append(first)

    if len(good_matches) < config.min_good_matches:
        raise AlignmentError("alignment: good matches below configured threshold")

    # ── 4) RANSAC 估计单应性矩阵 ──
    # 将匹配点坐标提取为 N×1×2 的 float32 数组
    scan_points = np.array(
        [scan_keypoints[m.queryIdx].pt for m in good_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    template_points = np.array(
        [template_keypoints[m.trainIdx].pt for m in good_matches],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    # cv2.findHomography + RANSAC：重投影误差阈值 config.ransac_reprojection_threshold px
    homography, inlier_mask = cv2.findHomography(
        scan_points,
        template_points,
        cv2.RANSAC,
        config.ransac_reprojection_threshold,
    )

    if homography is None or not np.isfinite(homography).all() or inlier_mask is None:
        raise AlignmentError("alignment: homography computation failed")

    # ── 5) 透视变换 —— 将扫描件 warp 到模板坐标系 ──
    # dsize=(template_width, template_height)：输出尺寸 = 模板尺寸
    # borderValue=(255,255,255)：空白区域填白
    template_height, template_width = template_bgr.shape[:2]
    aligned = cv2.warpPerspective(
        scan_bgr,
        homography,
        (template_width, template_height),
        flags=cv2.INTER_LINEAR,         # 双线性插值，平滑
        borderMode=cv2.BORDER_CONSTANT,  # 边界常量填充
        borderValue=(255, 255, 255),     # 填白色
    )

    # 内点比例：RANSAC 最终认可的正确匹配占比
    inlier_ratio = float(inlier_mask.ravel().sum() / len(good_matches))
    return AlignmentResult(
        aligned_bgr=aligned,
        good_matches=len(good_matches),
        inlier_ratio=inlier_ratio,
        homography=homography,
    )
