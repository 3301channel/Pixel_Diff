"""Pixel-Diff 公共 API。

通过懒加载（__getattr__）暴露 engine 模块，避免 import 时触发
PDF 渲染等重型依赖。只有实际访问 PixelDiffEngine 或 compare 时才导入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# 数据模型和异常总是直接可用，无需懒加载
from pixel_diff.exceptions import (
    AlignmentError,
    ConfigurationError,
    DimensionMismatchError,
    InputError,
    OutputError,
    PixelDiffError,
)
from pixel_diff.models import DifferenceRegion, PixelDiffConfig, PixelDiffResult

# 引擎类型仅在类型检查时导入，运行时通过 __getattr__ 懒加载
if TYPE_CHECKING:
    from pixel_diff.engine import PixelDiffEngine, compare

__all__ = [
    "AlignmentError",
    "ConfigurationError",
    "DifferenceRegion",
    "DimensionMismatchError",
    "InputError",
    "OutputError",
    "PixelDiffConfig",
    "PixelDiffEngine",
    "PixelDiffError",
    "PixelDiffResult",
    "compare",
]


def __getattr__(name: str) -> Any:
    """懒加载引擎 API，避免在 import pixel_diff 时触发 PDF 渲染和 OpenCV 依赖。

    仅当代码实际访问 PixelDiffEngine 或 compare 时才导入 engine 模块。
    """
    if name in {"PixelDiffEngine", "compare"}:
        from pixel_diff.engine import PixelDiffEngine, compare

        values = {"PixelDiffEngine": PixelDiffEngine, "compare": compare}
        return values[name]
    raise AttributeError(f"module 'pixel_diff' has no attribute {name!r}")
