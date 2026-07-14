"""HSV 色彩空间过滤 —— 去除扫描件上的红章蓝签。

将彩色扫描件转到 HSV 空间，用预设的 HSV 范围定位红色（公章）
和蓝色（手写签名）区域，将其填充为白色（255,255,255），避免
后续 XOR 差异检测误报。

注意：此操作会牺牲被印章/签名覆盖区域的差异检测能力，属于
有意为之的权衡——被覆盖的内容人工也无法辨识。
"""

from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.models import PixelDiffConfig

# OpenCV 中白色对应的 BGR 三通道值
_WHITE_BGR = (255, 255, 255)


def remove_colored_marks_bgr(image_bgr: np.ndarray, config: PixelDiffConfig) -> np.ndarray:
    """对 BGR 扫描件去除红章和蓝色签名，返回处理后的副本。

    仅在 config.filter_colored_marks 为 True 时执行过滤；
    否则原样返回副本（避免调用方需要条件判断）。

    处理流程：
    1. BGR → HSV 颜色空间
    2. 对红色 HSV 范围生成掩码（色环两端 0° 附近）
    3. 对蓝色 HSV 范围生成掩码
    4. 合并掩码，将命中像素填充为白色
    """
    if not config.filter_colored_marks:
        # 禁用色彩过滤时直接返回副本
        return image_bgr.copy()

    # BGR → HSV：OpenCV 的 HSV 中 H∈[0,179], S,V∈[0,255]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    # 红色在 HSV 色环的 0° 附近，对应 H 的两端
    #   H∈[0,  10]  — 正红偏橙
    #   H∈[170,180] — 正红偏紫
    for (h_low, s_low, v_low), (h_high, s_high, v_high) in config.red_hsv_ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, (h_low, s_low, v_low), (h_high, s_high, v_high)),
        )

    # 蓝色签名：H∈[100,124]，覆盖深蓝到浅蓝的手写墨水
    for (h_low, s_low, v_low), (h_high, s_high, v_high) in config.blue_hsv_ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, (h_low, s_low, v_low), (h_high, s_high, v_high)),
        )

    # 对命中区域填充白色，等同于"擦除"
    result = image_bgr.copy()
    result[mask > 0] = _WHITE_BGR
    return result
