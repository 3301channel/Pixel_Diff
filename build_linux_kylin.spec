# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller Linux 打包脚本 —— 生成 ELF 可执行文件夹（银河麒麟 aarch64 / x86_64 通用）。

产物：
    dist/pd_pyinstaller_build/
        compare      # CLI 比对工具
        run_api      # FastAPI 服务
        configs/     # 配置文件
        _internal/   # Python 运行时与依赖库

用法（麒麟 V10 终端）：
    source .venv/bin/activate
    export LD_LIBRARY_PATH=/opt/python3.12/lib
    pyinstaller --clean -y build_linux_kylin.spec
"""

from pathlib import Path

ROOT = Path(SPECPATH).resolve()
SRC = ROOT / "src"
CONFIGS = ROOT / "configs"
DISTPATH = ROOT / "dist"

# Linux 上不需要 win32com（Windows 专属）
HIDDEN_IMPORTS = [
    # pixel_diff 核心包
    "pixel_diff",
    "pixel_diff._app_paths",
    "pixel_diff.engine",
    "pixel_diff.models",
    "pixel_diff.exceptions",
    "pixel_diff.io",
    "pixel_diff.alignment",
    "pixel_diff.text_anchor_alignment",
    "pixel_diff.line_alignment",
    "pixel_diff.line_affine_alignment",
    "pixel_diff.line_piecewise_alignment",
    "pixel_diff.rigid_text_block_alignment",
    "pixel_diff.residual_line_alignment",
    "pixel_diff.local_warp",
    "pixel_diff.color_filter",
    "pixel_diff.text_layer",
    "pixel_diff.binarization",
    "pixel_diff.differ",
    "pixel_diff.morphology",
    "pixel_diff.regions",
    "pixel_diff.filter_pipeline",
    "pixel_diff.similarity",
    "pixel_diff.risk_review",
    "pixel_diff.change_classification",
    "pixel_diff.displacement",
    "pixel_diff.patch_export",
    "pixel_diff.visualization",
    "pixel_diff.report",
    "pixel_diff.timing",
    "pixel_diff.region_utils",
    "pixel_diff.logging_setup",
    "PIL",
    # pixel_diff_api 包
    "pixel_diff_api",
    "pixel_diff_api.app",
    "pixel_diff_api.settings",
    "pixel_diff_api.system_info",
    "pixel_diff_api.task_service",
    "pixel_diff_api.viewer",
]

# Linux 可用库保留，仅排除真的不需要的
EXCLUDES = [
    "rapidocr_onnxruntime",
    "onnxruntime",
    "easyocr",
    "pytesseract",
    "matplotlib",
    "scipy",
    "tkinter",
    # PIL 保留（visualization 顶层依赖）
    "torch",
    "tensorflow",
    "pywin32",        # Windows 专属
    "win32com",       # Windows 专属
    "pythoncom",      # Windows 专属
]

DATAS = [(str(CONFIGS), "configs")]

a_cli = Analysis(
    [str(ROOT / "scripts" / "compare.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

a_api = Analysis(
    [str(ROOT / "scripts" / "run_api.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(
    a_cli.pure + a_api.pure,
    a_cli.zipped_data + a_api.zipped_data,
)

exe_cli = EXE(
    pyz,
    a_cli.scripts,
    [],
    exclude_binaries=True,
    name="compare",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

exe_api = EXE(
    pyz,
    a_api.scripts,
    [],
    exclude_binaries=True,
    name="run_api",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe_cli,
    exe_api,
    a_cli.binaries,
    a_api.binaries,
    a_cli.datas,
    a_api.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="pd_pyinstaller_build",
)

# 把 configs/ 额外拷贝到 exe 同级（便于查看/修改）
import shutil

_dist_configs = DISTPATH / "pd_pyinstaller_build" / "configs"
shutil.copytree(CONFIGS, _dist_configs, dirs_exist_ok=True)
