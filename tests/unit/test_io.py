from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pixel_diff import InputError, OutputError
from pixel_diff.io import (
    convert_docx_to_pdf,
    get_document_page_count,
    load_document_page_bgr,
    read_image_bgr,
    render_pdf_page_bgr,
    validate_image_array,
    write_image_bgr,
    write_json,
)


def test_read_write_image_and_json_round_trip(tmp_path: Path) -> None:
    image = np.full((8, 9, 3), 255, dtype=np.uint8)
    image[2, 3] = (0, 0, 0)
    image_path = tmp_path / "nested" / "image.png"
    json_path = tmp_path / "result.json"

    assert write_image_bgr(image_path, image) == str(image_path)
    loaded = read_image_bgr(image_path)
    assert loaded.shape == image.shape

    assert write_json(json_path, {"ok": True}) == str(json_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}


def test_load_document_page_rejects_missing_and_unsupported_paths(tmp_path: Path) -> None:
    with pytest.raises(InputError):
        load_document_page_bgr(tmp_path / "missing.pdf", page=0, dpi=300)

    text_path = tmp_path / "sample.txt"
    text_path.write_text("not an image", encoding="utf-8")
    with pytest.raises(InputError):
        load_document_page_bgr(text_path, page=0, dpi=300)


def test_read_image_rejects_damaged_file(tmp_path: Path) -> None:
    damaged = tmp_path / "damaged.png"
    damaged.write_bytes(b"not-image")

    with pytest.raises(InputError):
        read_image_bgr(damaged)


def test_validate_image_array_rejects_bad_arrays() -> None:
    validate_image_array(np.zeros((2, 2), dtype=np.uint8), "gray")
    validate_image_array(np.zeros((2, 2, 3), dtype=np.uint8), "bgr")

    with pytest.raises(InputError):
        validate_image_array(np.zeros((2, 2), dtype=np.float32), "float")
    with pytest.raises(InputError):
        validate_image_array(np.zeros((2, 2, 4), dtype=np.uint8), "rgba")


def test_render_pdf_page_and_page_bounds(tmp_path: Path) -> None:
    fitz = _fitz_or_skip()
    pdf_path = tmp_path / "one_page.pdf"
    document = fitz.open()
    page = document.new_page(width=72, height=72)
    page.insert_text((10, 30), "PDF")
    document.save(pdf_path)
    document.close()

    image = render_pdf_page_bgr(pdf_path, page=0, dpi=72)
    assert image.shape[:2] == (72, 72)

    with pytest.raises(InputError):
        render_pdf_page_bgr(pdf_path, page=2, dpi=72)


def test_get_document_page_count_for_pdf_and_image(tmp_path: Path) -> None:
    fitz = _fitz_or_skip()
    pdf_path = tmp_path / "two_pages.pdf"
    document = fitz.open()
    document.new_page(width=72, height=72)
    document.new_page(width=72, height=72)
    document.save(pdf_path)
    document.close()

    image_path = tmp_path / "page.png"
    cv2.imwrite(str(image_path), np.zeros((3, 4, 3), dtype=np.uint8))

    assert get_document_page_count(pdf_path) == 2
    assert get_document_page_count(image_path) == 1


def test_write_image_raises_output_error_for_invalid_extension(tmp_path: Path) -> None:
    with pytest.raises(OutputError):
        write_image_bgr(tmp_path / "image.badext", np.zeros((2, 2, 3), dtype=np.uint8))


def test_load_document_page_reads_image_suffix(tmp_path: Path) -> None:
    path = tmp_path / "page.png"
    cv2.imwrite(str(path), np.zeros((3, 4, 3), dtype=np.uint8))

    assert load_document_page_bgr(path, page=0, dpi=300).shape == (3, 4, 3)


def test_convert_docx_rejects_non_docx_path(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("not docx", encoding="utf-8")

    with pytest.raises(InputError):
        convert_docx_to_pdf(path, tmp_path / "out")


def _fitz_or_skip() -> object:
    try:
        import fitz
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PyMuPDF unavailable: {exc}")
    return fitz
