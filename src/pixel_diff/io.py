"""文档加载渲染、图像读写、DOCX→PDF 转换。

输入层负责将各类格式的输入文件统一转为 OpenCV BGR uint8 图像：
- PDF   → pypdfium2 渲染（优先），PyMuPDF 回退
- 图片  → cv2.imdecode + np.fromfile（兼容中文路径）
- DOCX  → LibreOffice headless 转 PDF（优先），MS Word COM 回退

输出层提供 BGR 图像写入和 JSON 写入。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from pixel_diff.exceptions import InputError, OutputError

logger = logging.getLogger(__name__)

# LibreOffice 常用安装路径（Windows）
_LO_PATHS = (
    "C:\\Program Files\\LibreOffice\\program\\soffice.exe",
    "C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe",
)


def load_document_page_bgr(
    path: str | Path,
    page: int,
    dpi: int,
) -> np.ndarray:
    """加载文档指定页面，返回 BGR uint8 图像（300 DPI）。

    根据文件扩展名自动选择加载方式：
    - .pdf  → _render_pdf()
    - .jpg/.jpeg/.png/.tiff/.tif/.bmp → _read_image_bgr()
    - 其他  → 抛出 InputError
    """
    file = Path(path)
    if not file.exists():
        raise InputError(f"input: path does not exist: {file}")
    suffix = file.suffix.lower()
    if suffix == ".pdf":
        return _render_pdf(file, page, dpi)
    if suffix in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"):
        return _read_image_bgr(file)
    raise InputError(f"input: unsupported file type {suffix!r}")


def get_document_page_count(path: str | Path) -> int:
    """获取文档总页数。PDF 用 pypdfium2，图片固定为 1 页。"""
    file = Path(path)
    if file.suffix.lower() == ".pdf":
        # pypdfium2 在非 Windows 平台存在类型问题，通过 try/except 获取 API
        try:
            import pypdfium2

            pdf = pypdfium2.PdfDocument(str(file))
            return int(pdf.get_page_count())
        except Exception:
            import fitz

            return cast(int, fitz.open(file).page_count)
    return 1


def validate_image_array(image: np.ndarray, label: str) -> None:
    """校验图像数组为合法的 BGR uint8 格式。

    Raises:
        InputError: 图像维度、通道数或数据类型不符合预期。
    """
    if image.ndim not in (2, 3):
        raise InputError(f"input: {label} image must be 2D or 3D array")
    if image.ndim == 3 and image.shape[2] != 3:
        raise InputError(f"input: {label} image must have 3 channels")
    if image.dtype != np.uint8:
        raise InputError(f"input: {label} image must be uint8")


def write_image_bgr(output_path: str | Path, image_bgr: np.ndarray) -> str:
    """将 BGR 图像写入磁盘（PNG 格式）。

    返回写入的绝对路径字符串。
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = cv2.imwrite(str(out), image_bgr)
    except cv2.error as exc:
        raise OutputError(f"output: failed to write image to {out}") from exc
    if not written:
        raise OutputError(f"output: failed to write image to {out}")
    return str(out.resolve())


def write_json(output_path: str | Path, data: dict[str, Any]) -> str:
    """将字典写入 JSON 文件（UTF-8，indent=2，ensure_ascii=False）。

    返回写入的绝对路径字符串。
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return str(out.resolve())


def convert_docx_to_pdf(docx_path: Path, output_dir: Path) -> Path:
    """将 DOCX 文件转为 PDF。

    优先使用 LibreOffice headless 模式（跨平台），
    失败则回退到 Microsoft Word COM 自动化（仅 Windows）。
    """
    if docx_path.suffix.lower() != ".docx":
        raise InputError(f"input: unsupported file format: {docx_path.suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    soffice = _find_libreoffice()
    if soffice is not None:
        return _convert_via_libreoffice(docx_path, output_dir, soffice)
    if sys.platform == "win32":
        return _convert_via_word_com(docx_path, output_dir)
    raise InputError("input: no usable DOCX→PDF converter found")


def _find_libreoffice() -> str | None:
    """在 PATH 和常见安装路径中查找 LibreOffice 可执行文件。"""
    if sys.platform == "win32":
        # Windows：优先命令行查找，再遍历安装路径
        if (
            subprocess.call(
                ["where", "soffice"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            == 0
        ):
            return "soffice"
        for candidate in _LO_PATHS:
            if Path(candidate).exists():
                return candidate
        return None
    # Linux/macOS：依赖 PATH
    if (
        subprocess.call(
            ["which", "soffice"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    ):
        return "soffice"
    return None


def _render_pdf(pdf_path: Path, page: int, dpi: int) -> np.ndarray:
    """渲染 PDF 指定页为白底 BGR uint8 图像。

    优先使用 pypdfium2（快速、轻量），失败则回退到 PyMuPDF（功能全面但更重）。
    注意：pypdfium2 的 libpdfium 在 PyInstaller 冻结态下多线程并行渲染会 SIGSEGV，
    因此多线程报告模式通过 ``PD_DISABLE_PYPDFIUM2=1`` 禁用 pypdfium2，统一走
    线程安全的 PyMuPDF。
    """
    if os.environ.get("PD_DISABLE_PYPDFIUM2") != "1":
        try:
            return _render_pdf_pypdfium2(pdf_path, page, dpi)
        except Exception:
            logger.info("pypdfium2 failed, falling back to PyMuPDF")
    return _render_pdf_pymupdf(pdf_path, page, dpi)


def render_pdf_page_bgr(path: str | Path, page: int, dpi: int) -> np.ndarray:
    """Render one PDF page as OpenCV BGR uint8."""

    return _render_pdf(Path(path), page, dpi)


def _render_pdf_pypdfium2(pdf_path: Path, page: int, dpi: int) -> np.ndarray:
    """使用 pypdfium2 渲染 PDF 页面。

    pypdfium2 直接调用 PDFium（Chromium 的 PDF 引擎），
    渲染速度快、内存占用低。新版 API 用 ``page.render()``（旧版
    ``PdfDocument.render`` 在 pypdfium2 5.x 已移除，导致一直静默
    fallback 到 PyMuPDF）。
    """
    import pypdfium2

    scale = dpi / 72.0  # PDF 内部使用 72 DPI，需放大到目标 DPI
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    bitmap = pdf[page].render(scale=scale)
    arr = bitmap.to_numpy()
    # pypdfium2 5.x 默认输出 RGB（3 通道）；若为 RGBA（4 通道）则补白底转 BGR
    if arr.shape[2] == 4:
        return _rgba_to_bgr_white_bg(arr, scale)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _render_pdf_pymupdf(pdf_path: Path, page: int, dpi: int) -> np.ndarray:
    """使用 PyMuPDF (fitz) 渲染 PDF 页面。

    作为 pypdfium2 的回退方案，兼容性更广。
    """
    import fitz

    doc = fitz.open(pdf_path)
    if page < 0 or page >= doc.page_count:
        doc.close()
        raise InputError(f"input: page index out of range: {page}")
    page_obj = doc[page]
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)
    pix = page_obj.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
    raw = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    # 处理不同通道数：RGBA→BGR 补白底，RGB→BGR
    if raw.shape[2] == 4:
        return _rgba_to_bgr_white_bg(raw, scale)
    return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)


def _rgba_to_bgr_white_bg(rgba: np.ndarray, scale: float) -> np.ndarray:
    """将 RGBA 图像合成为白底 BGR。

    渲染引擎输出的 RGBA 中，透明区域 A=0，需要补白色背景。
    合成公式：out = foreground * (A/255) + white * (1 - A/255)
    """
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    alpha = np.expand_dims(alpha, axis=2)  # (H, W) → (H, W, 1)
    white_bg = np.full_like(bgr, 255, dtype=np.float32)
    return (bgr.astype(np.float32) * alpha + white_bg * (1.0 - alpha)).astype(np.uint8)


def _read_image_bgr(path: Path) -> np.ndarray:
    """读取任意图片文件为 BGR uint8。

    使用 np.fromfile + cv2.imdecode 避免 OpenCV 中文路径问题。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise InputError(f"input: failed to decode image {path}")
    return img


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read an image file as OpenCV BGR uint8."""

    return _read_image_bgr(Path(path))


def _convert_via_libreoffice(docx_path: Path, output_dir: Path, soffice: str) -> Path:
    """使用 LibreOffice headless 模式将 DOCX 转为 PDF。

    输出到 --outdir 指定目录，文件名与源文件相同（扩展名改为 .pdf）。
    timeout=120s 应对大型文档。
    """
    subprocess.run(
        [
            soffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(docx_path),
        ],
        check=True,
        timeout=120,
    )
    pdf_path = output_dir / f"{docx_path.stem}.pdf"
    if not pdf_path.exists():
        raise InputError(f"input: LibreOffice conversion produced no PDF at {pdf_path}")
    return pdf_path


def _convert_via_word_com(docx_path: Path, output_dir: Path) -> Path:
    """使用 Microsoft Word COM 自动化将 DOCX 转为 PDF（仅 Windows）。

    作为 LibreOffice 不可用时的回退方案。
    """
    import pythoncom
    import win32com.client as win32

    pythoncom.CoInitialize()
    try:
        word = win32.Dispatch("Word.Application")
        word.Visible = False
        try:
            doc = word.Documents.Open(str(docx_path.resolve()))
            pdf_path = output_dir / f"{docx_path.stem}.pdf"
            doc.SaveAs2(str(pdf_path.resolve()), FileFormat=17)  # 17 = wdFormatPDF
            doc.Close()
            return pdf_path
        finally:
            word.Quit()
    finally:
        pythoncom.CoUninitialize()
