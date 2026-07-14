"""Pixel-Diff 领域异常定义。

所有异常继承自 PixelDiffError，按错误来源细分：
- InputError        — 输入文件/路径/页面无效
- ConfigurationError — 算法参数配置不合法
- AlignmentError     — SIFT/RANSAC 配准无法完成
- DimensionMismatchError — 两张图尺寸不一致，无法逐像素操作
- OutputError        — 结果文件写入失败
"""


class PixelDiffError(Exception):
    """Pixel-Diff 所有领域异常的基类。"""


class InputError(PixelDiffError):
    """输入路径、文件、页面或解码图像无效时抛出。"""


class ConfigurationError(PixelDiffError):
    """算法配置参数不合法时抛出（如阈值越界、核尺寸非法）。"""


class AlignmentError(PixelDiffError):
    """全局图像配准无法安全计算时抛出（特征点不足、单应性计算失败等）。"""


class DimensionMismatchError(PixelDiffError):
    """两张进入逐像素操作的图像尺寸不一致时抛出。"""


class OutputError(PixelDiffError):
    """JSON 或可视化结果写入磁盘失败时抛出。"""
