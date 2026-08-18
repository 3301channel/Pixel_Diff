"""形态学清理 —— 对 XOR 差异掩码做去噪和碎片合并。

差异掩码通常包含大量微小噪声点和碎片化的差异块，需要在
提取轮廓之前经过形态学操作处理：

1. remove_small_components() — 删除面积 < 12px² 的小连通域
2. 开运算 (MORPH_OPEN)      — 断开弱连接、去除细毛刺
3. 闭运算 (MORPH_CLOSE)     — 填充小孔洞、弥合近邻碎片
4. 膨胀 (dilate)            — 将碎片化的差异合并为连续区域块
"""

from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.models import PixelDiffConfig


def clean_difference_mask(diff_binary: np.ndarray, config: PixelDiffConfig) -> np.ndarray:
    """对 XOR 差异掩码执行完整的形态学清理管道。

    参数化：
    - min_noise_component_area=12  — 小于此面积的连通域视为噪声
    - open_kernel=(3,3), iterations=1 — 开运算核与次数
    - close_kernel=(3,3), iterations=1 — 闭运算核与次数
    - dilate_kernel=(15,10), iterations=1 — 膨胀核（横向 > 纵向，适配横排中文文本）
    """
    result = diff_binary.copy()

    # 1) 去小连通域 —— 面积 < 12px² 的孤立白点直接删除
    result = remove_small_components(result, config.min_noise_component_area)

    # 2) 开运算（先腐蚀后膨胀）—— 去掉细毛刺、断开弱连接
    if config.morph_iterations_open:
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_OPEN,
            _kernel(config.open_kernel),           # 矩形结构元素 (3,3)
            iterations=config.morph_iterations_open, # 1 次
        )

    # 3) 闭运算（先膨胀后腐蚀）—— 填充区域内的小孔洞
    if config.morph_iterations_close:
        result = cv2.morphologyEx(
            result,
            cv2.MORPH_CLOSE,
            _kernel(config.close_kernel),           # 矩形结构元素 (3,3)
            iterations=config.morph_iterations_close, # 1 次
        )

    # 4) 膨胀 —— 将同一文字行的碎片差异横向合并
    #    核 (15,10) 横向 > 纵向，适配横排中文的分布特征
    if config.morph_iterations_dilate:
        result = cv2.dilate(
            result,
            _kernel(config.dilate_kernel),          # 矩形结构元素 (15,10)
            iterations=config.morph_iterations_dilate, # 1 次
        )
    return result


def prepare_difference_masks(
    diff_binary: np.ndarray,
    unchanged_text_mask: np.ndarray | None,
    config: PixelDiffConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return detection and recall masks without cleaning the same input twice."""

    recall_mask = clean_difference_mask(diff_binary, config)
    if unchanged_text_mask is None:
        return recall_mask, recall_mask
    detection_mask = clean_difference_mask(
        diff_binary & unchanged_text_mask,
        config,
    )
    return detection_mask, recall_mask


def remove_small_components(diff_binary: np.ndarray, min_area: float) -> np.ndarray:
    """删除面积 < min_area 的前景连通域。

    使用连通组件分析 (8-连通)，遍历每个组件，
    面积不达标的直接丢弃。适用于去除 XOR 后
    的散点噪声（灰尘、微小的配准残余像素）。
    """
    if min_area <= 0:
        return diff_binary.copy()

    # 连通组件分析，8-连通（含对角线）
    _component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        diff_binary,
        connectivity=8,
    )

    keep = _component_keep_lookup(stats, min_area)
    return np.asarray(keep[labels], dtype=np.uint8) * 255


def _component_keep_lookup(stats: np.ndarray, min_area: float) -> np.ndarray:
    """Build a label-indexed keep table with background always disabled."""

    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep


def _kernel(size: tuple[int, int]) -> np.ndarray:
    """生成矩形结构元素，用于形态学操作。"""
    return cv2.getStructuringElement(cv2.MORPH_RECT, size)
