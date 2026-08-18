# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for bundling Pixel-Diff as a Linux executable.

Pillow must be collected explicitly because its plugins are loaded lazily
at runtime; without collect_all("PIL") the frozen bundle fails to import
image backends. pixel_diff / pixel_diff_api submodules are collected
explicitly so dynamically imported modules are not missed. configs/ is
bundled as data (frozen runtime falls back to exe-dir/_internal/configs).
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

pillow_datas, pillow_binaries, pillow_hiddenimports = collect_all("PIL")

hiddenimports = (
    collect_submodules("pixel_diff")
    + collect_submodules("pixel_diff_api")
    + pillow_hiddenimports
)

a = Analysis(
    ["scripts/compare.py"],
    pathex=["src"],
    binaries=pillow_binaries,
    datas=pillow_datas + [("configs", "configs")],
    hiddenimports=hiddenimports,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pixel_diff",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="pixel_diff",
)
