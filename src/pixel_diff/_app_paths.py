"""冻结态 / 开发态通用的项目路径解析。

打包（PyInstaller onedir 或 Nuitka standalone）后，``__file__`` 指向的位置
与源码树不同，原来的 ``Path(__file__).resolve().parents[N]`` 路径会失效。
本模块统一收口所有路径解析，兼容两种打包方式：

- ``is_frozen()``     是否处于冻结运行态（PyInstaller / Nuitka 均会置位）
- ``app_root()``      项目根目录（开发态=仓库根；冻结态=exe 所在目录）
- ``resource_path()`` 相对项目根的资源路径（如 ``configs/default.yaml``），
                     在 exe 同级目录与 ``_internal`` / ``_MEIPASS`` 间自动回退
- ``compare_entry()`` CLI 比对入口（冻结态=同级 ``compare.exe``；
                     开发态=``scripts/compare.py``）

构建时 ``configs/`` 作为数据文件放在 exe 同级目录（PyInstaller 由 spec 拷贝，
Nuitka 由 --include-data-dir 捆绑），故两种状态下都能定位到配置。
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """是否处于 PyInstaller / Nuitka 冻结运行态。"""
    return bool(getattr(sys, "frozen", False))


def _frozen_candidates(exe_dir: Path) -> list[Path]:
    """冻结态下可能的资源根目录候选项（按顺序回退）。

    - PyInstaller onedir：数据文件在 ``_internal/``
    - PyInstaller onefile：数据文件解压到 ``sys._MEIPASS``
    - Nuitka standalone：数据文件在 exe 同级目录
    - 构建脚本也会把 configs 额外拷贝到 exe 同级目录（便于查看/修改）
    """
    cands = [exe_dir, exe_dir / "_internal"]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands.append(Path(meipass))
    return cands


def app_root() -> Path:
    """项目根目录（用于写输出、定位配置等）。

    开发态：本文件位于 ``src/pixel_diff/_app_paths.py``，向上两级即仓库根。
    冻结态（onedir / standalone）：exe 所在目录即用户可操作的根，
    运行时产物（artifacts/）也放在这里，便于查看。
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def resource_path(rel_path: str | Path) -> Path:
    """返回相对项目根的资源绝对路径，自动在候选目录间回退。"""
    rel_path = Path(rel_path)
    if is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        for base in _frozen_candidates(exe_dir):
            cand = base / rel_path
            if cand.exists():
                return cand
        return exe_dir / rel_path
    return app_root() / rel_path


def compare_entry() -> Path:
    """CLI 比对入口路径。

    冻结态为 exe 同级的 ``compare.exe``；开发态为 ``scripts/compare.py``。
    """
    if is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        platform_name = "compare.exe" if sys.platform == "win32" else "compare"
        fallback_name = "compare" if platform_name == "compare.exe" else "compare.exe"
        candidates = (exe_dir / platform_name, exe_dir / fallback_name)
        return next((path for path in candidates if path.is_file()), candidates[0])
    return app_root() / "scripts" / "compare.py"
