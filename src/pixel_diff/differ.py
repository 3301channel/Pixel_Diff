"""逐像素 XOR 异或差异检测与边缘裁剪。

差异检测阶段的核心操作：
1. xor_difference() — 两幅二值图逐位异或，255=有差异，0=无差异
2. crop_edges()    — 裁掉四周边缘伪影（扫描仪黑边、装订孔等）
"""

from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.exceptions import DimensionMismatchError
from pixel_diff.models import PixelDiffConfig


def xor_difference(
    scan_binary: np.ndarray,
    template_binary: np.ndarray,
    dilate_radius: int = 0,
) -> np.ndarray:
    """对两幅二值图逐像素异或，返回差异掩码。

    前提：两幅图已通过 warpPerspective 对齐到相同尺寸。
    约定：前景=0（黑），背景=255（白），XOR 结果中 255 表示差异。

    异或真值表：
        扫描件  XOR  模板  =  结果（含义）
         0（字）       0（字） =  0  无差异
       255（空）     255（空） =  0  无差异
         0（字）     255（空） = 255  疑似新增
       255（空）       0（字） = 255  疑似删除

    Args:
        dilate_radius: 膨胀半径。>0 时先对两幅二值图做相同膨胀，再异或。
            用于吸收扫描模糊/抗锯齿/轻微错位造成的亚像素边缘差异。
    """
    if scan_binary.shape != template_binary.shape:
        raise DimensionMismatchError(
            "difference: scan and template binary images must have identical dimensions"
        )
    if dilate_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (2 * dilate_radius + 1, 2 * dilate_radius + 1)
        )
        scan_dilated = cv2.dilate(scan_binary, kernel)
        template_dilated = cv2.dilate(template_binary, kernel)
        return cv2.bitwise_xor(scan_dilated, template_dilated)
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


def mask_missing_table_lines(
    diff_binary: np.ndarray,
    scan_binary: np.ndarray,
    template_binary: np.ndarray,
    config: PixelDiffConfig,
) -> np.ndarray:
    """屏蔽「模板有表格线、扫描件对应位置缺失表格线」造成的差异。

    电子版模板通常带有清晰的表格边框线，而扫描件/重新渲染的版本
    可能整体丢失这些表格线（打印/扫描褪色、转 PDF 丢失矢量线等）。
    这类差异属于版式差异而非内容修改，会在大范围内产生细长的
    假阳性（尤其表格密集的文档）。

    处理逻辑（模板-only，对扫描噪声鲁棒）：
    1. 仅在清晰模板二值图中提取长横线、长竖线（真正的表格线）。
    2. 把模板表格线位置的差异像素从差异掩码中屏蔽（置 0）。

    之所以不在扫描件中检测表格线做"缺失比例"判定，是因为模糊扫描
    件中的文字笔画粘连很容易产生伪长线，导致比例判定失效。模板
    线条清晰可靠，直接以其位置作为"结构差异"的屏蔽依据更稳定。

    注意：此过滤器仅在 ``table_line_filter_enabled=True`` 时生效；
    如果业务需要把"扫描件缺失表格线"视为真实内容修改，可关闭它。
    """
    if not config.table_line_filter_enabled or config.table_line_min_length <= 0:
        return diff_binary
    if diff_binary.shape != template_binary.shape:
        return diff_binary

    # 前景=0（黑），背景=255（白）。转成 前景=1 便于形态学。
    template_fg = (template_binary == 0).astype(np.uint8)

    min_len = int(config.table_line_min_length)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))

    template_lines_horiz = cv2.morphologyEx(template_fg, cv2.MORPH_OPEN, horiz_kernel)
    template_lines_vert = cv2.morphologyEx(template_fg, cv2.MORPH_OPEN, vert_kernel)
    template_lines = cv2.bitwise_or(template_lines_horiz, template_lines_vert)

    if cv2.countNonZero(template_lines) == 0:
        return diff_binary

    # 把模板表格线位置上的差异像素屏蔽
    mask = template_lines.astype(np.uint8) * 255
    return cv2.bitwise_and(diff_binary, cv2.bitwise_not(mask))
