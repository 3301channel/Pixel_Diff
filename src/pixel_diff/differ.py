"""逐像素 XOR 异或差异检测与边缘裁剪。

差异检测阶段的核心操作：
1. xor_difference() — 两幅二值图逐位异或，255=有差异，0=无差异
2. crop_edges()    — 裁掉四周边缘伪影（扫描仪黑边、装订孔等）
"""

from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.exceptions import DimensionMismatchError


def xor_difference(scan_binary: np.ndarray, template_binary: np.ndarray) -> np.ndarray:
    """对两幅二值图逐像素异或，返回差异掩码。

    前提：两幅图已通过 warpPerspective 对齐到相同尺寸。
    约定：前景=0（黑），背景=255（白），XOR 结果中 255 表示差异。

    异或真值表：
        扫描件  XOR  模板  =  结果（含义）
         0（字）       0（字） =  0  无差异
       255（空）     255（空） =  0  无差异
         0（字）     255（空） = 255  疑似新增
       255（空）       0（字） = 255  疑似删除
    """
    if scan_binary.shape != template_binary.shape:
        raise DimensionMismatchError(
            "difference: scan and template binary images must have identical dimensions"
        )
    return cv2.bitwise_xor(scan_binary, template_binary)


def crop_edges(diff_binary: np.ndarray, margin: int) -> np.ndarray:
    """将差异掩码四周 margin 像素宽的区域置零（当作无差异）。

    目的：抑制扫描件边框区域常见的伪影：
    - 扫描仪玻璃边缘反射产生的黑边/亮边
    - 纸张装订孔、打孔印记
    - 扫描件盖板漏光区域

    这些区域扫描件有、模板没有，XOR 后会产生大片假阳性差异。
    默认 margin=40，在 300 DPI 下约 3.4mm。
    """
    cropped = diff_binary.copy()
    if margin == 0:
        return cropped
    # 上下左右各裁 margin 像素，置零（0=无差异）
    cropped[:margin, :] = 0
    cropped[-margin:, :] = 0
    cropped[:, :margin] = 0
    cropped[:, -margin:] = 0
    return cropped
