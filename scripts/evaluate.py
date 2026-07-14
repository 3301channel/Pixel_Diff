"""占位评估脚本 —— 用于量化算法在冻结标准数据集上的表现。

当前为占位实现，尚未集成具体的标注真值（ground-truth boxes）。
未来扩展方向：
- 读取 manifest 文件中定义的测试用例
- 加载每个用例的标注真值（人工标注的差异区域坐标）
- 运行 Pixel-Diff 比对
- 计算精确率/召回率/F1-score
- 输出评估报告

用法:
    python -m scripts.evaluate <manifest.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Placeholder evaluator for frozen acceptance datasets."
    )
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    raise SystemExit(
        "Acceptance evaluation requires a frozen manifest with ground-truth boxes; "
        f"not found/implemented for {args.manifest}. "
        + json.dumps({"status": "not_ready"}, ensure_ascii=False)
    )


if __name__ == "__main__":
    raise SystemExit(main())
