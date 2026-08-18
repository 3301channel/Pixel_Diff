"""二值化 —— 将灰度/BGR 图像转为 0/255 二值图。

扫描件和模板采用**不同策略**，因为它们的物理来源不同：

扫描件（binarize_scan_bgr）：
  扫描仪光照不均 + 传感器噪声 + 纸张纹理
  → 先用双边滤波保边去噪 + 中值滤波去椒盐噪声
  → 再用自适应高斯阈值应对局部光照变化

模板（binarize_template_bgr）：
  电子 PDF 直接渲染，纯白背景 + 纯黑文字，无噪声无光照不均
  → 直接用 Otsu 全局阈值即可

约定：前景（文字） = 0（黑），背景（空白） = 255（白）
这是有意设计的反向约定，使得 XOR 异或时：
  有字(0) XOR 无字(255) = 255  → 差异信号
  无字(255) XOR 有字(0) = 255  → 差异信号
  相同(0 vs 0 或 255 vs 255) = 0  → 无差异
"""

from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.models import PixelDiffConfig


def binarize_scan_bgr(
    image_bgr: np.ndarray,
    config: PixelDiffConfig,
    *,
    gray: np.ndarray | None = None,
) -> np.ndarray:
    """对扫描件 BGR 图像做三级去噪 + 自适应阈值二值化。

    处理链：
    1. BGR → 灰度
    2. 双边滤波 (d=9, σ_color=75, σ_space=75) — 保边去噪
    3. 中值滤波 (kernel=3) — 去椒盐噪声
    4. 自适应高斯阈值 (blockSize=21, C=10) — 应对光照不均

    返回：0=前景(文字), 255=背景
    """
    # 转灰度
    if gray is None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 双边滤波：空间域 + 灰度域双重高斯加权
    #   - 平坦区域（灰度相近）→ 平滑掉纸张纹理和扫描噪声
    #   - 边缘区域（灰度剧变）→ 保持锐利，不模糊文字笔画边界
    denoised = cv2.bilateralFilter(
        gray,
        config.bilateral_diameter,       # d=9，邻域直径
        config.bilateral_sigma_color,    # σ_color=75，灰度域标准差
        config.bilateral_sigma_space,    # σ_space=75，空间域标准差
    )

    # 中值滤波：取 3×3 邻域中位数，专门消除孤立的黑点/白点（椒盐噪声）
    # 不影响正常文字笔画（周围也多是黑像素）
    if config.median_blur_kernel:
        denoised = cv2.medianBlur(denoised, config.median_blur_kernel)

    # 自适应高斯阈值：
    #   - 对每个像素，取 21×21 邻域的高斯加权均值
    #   - 如果像素灰度 < 加权均值 - C(10)，判为前景（0，黑）
    #   - 否则判为背景（255，白）
    return cv2.adaptiveThreshold(
        denoised,
        255,  # maxValue
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 高斯加权
        cv2.THRESH_BINARY,               # 二值化模式
        config.adaptive_block_size,       # blockSize=21
        config.adaptive_c,               # C=10，从均值中减去的偏移
    )


def binarize_template_bgr(
    image_bgr: np.ndarray,
    *,
    gray: np.ndarray | None = None,
) -> np.ndarray:
    """对模板 BGR 图像做 Otsu 全局阈值二值化。

    模板是电子 PDF 渲染而成，质量好：
    - 背景均匀纯白（≈255）
    - 文字均匀纯黑（≈0）
    - 灰度直方图呈双峰分布

    Otsu 算法自动寻找最大化类间方差的阈值，在此场景下非常精准。

    返回：0=前景(文字), 255=背景
    """
    if gray is None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu：threshold=0 时自动计算最优阈值
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary
