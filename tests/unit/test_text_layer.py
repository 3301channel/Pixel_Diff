from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixel_diff import text_layer
from pixel_diff.models import DifferenceRegion, PixelDiffConfig


def test_sensitive_recall_is_disabled_by_default(tmp_path: Path) -> None:
    template = _write_template(tmp_path / "template.pdf")
    diff = np.full((500, 900), 255, dtype=np.uint8)

    regions = text_layer.extract_sensitive_text_recall_regions(
        template, page=0, diff_mask=diff, config=PixelDiffConfig()
    )

    assert regions == []


def test_sensitive_recall_merges_fragmented_digit_and_latin_signals(tmp_path: Path) -> None:
    template = _write_template(tmp_path / "template.pdf")
    rows = text_layer._extract_pdf_rows(template, page=0, dpi=300)
    diff = np.zeros((500, 900), dtype=np.uint8)
    sensitive_chars = [char for row in rows for char in row.chars if char.value in "62second"]
    for char in sensitive_chars[::2]:
        diff[char.y0 : char.y1 + 1, char.x0 : char.x1 + 1] = 255
    config = PixelDiffConfig(
        sensitive_text_recall_enabled=True,
        sensitive_text_recall_min_density=0.01,
    )

    regions = text_layer.extract_sensitive_text_recall_regions(
        template, page=0, diff_mask=diff, config=config
    )

    texts = [region.template_text or "" for region in regions]
    assert any("010-6528" in value for value in texts)
    assert any("second" in value for value in texts)


def test_sensitive_recall_rejects_empty_sensitive_rois(tmp_path: Path) -> None:
    template = _write_template(tmp_path / "template.pdf")
    config = PixelDiffConfig(
        sensitive_text_recall_enabled=True,
        sensitive_text_recall_min_density=0.01,
    )

    regions = text_layer.extract_sensitive_text_recall_regions(
        template,
        page=0,
        diff_mask=np.zeros((500, 900), dtype=np.uint8),
        config=config,
    )

    assert regions == []


def test_recalled_similarity_filter_removes_shifted_text_region() -> None:
    template = np.full((220, 500), 255, dtype=np.uint8)
    scan = template.copy()
    template[80:110, 120:210] = 0
    scan[80:110, 134:224] = 0
    region = DifferenceRegion(
        id=1,
        x=110,
        y=72,
        width=130,
        height=46,
        area=3000,
        template_text="contract",
    )
    config = PixelDiffConfig(
        sensitive_recall_similarity_filter_enabled=True,
        sensitive_recall_similarity_search_radius=16,
        sensitive_recall_similarity_iou_threshold=0.62,
    )

    result = text_layer.filter_recalled_similarity_regions(
        [region],
        scan_binary=scan,
        template_binary=template,
        config=config,
    )

    assert result == []


def test_recalled_similarity_filter_uses_wide_search_only_for_page_number() -> None:
    template = np.full((400, 500), 255, dtype=np.uint8)
    scan = template.copy()
    template[360:385, 220:235] = 0
    scan[360:385, 274:289] = 0
    region = DifferenceRegion(
        id=1,
        x=205,
        y=355,
        width=100,
        height=45,
        area=750,
        template_text=None,
    )
    config = PixelDiffConfig(
        sensitive_recall_similarity_filter_enabled=True,
        sensitive_recall_similarity_search_radius=16,
        sensitive_recall_page_number_search_radius=64,
        sensitive_recall_similarity_iou_threshold=0.62,
    )

    result = text_layer.filter_recalled_similarity_regions(
        [region],
        scan_binary=scan,
        template_binary=template,
        config=config,
        template_text_by_region={1: "3"},
    )

    assert result == []


def test_recalled_similarity_filter_keeps_real_change_and_can_be_disabled() -> None:
    template = np.full((220, 500), 255, dtype=np.uint8)
    scan = template.copy()
    template[80:110, 120:145] = 0
    scan[80:110, 185:225] = 0
    region = DifferenceRegion(
        id=1,
        x=110,
        y=72,
        width=130,
        height=46,
        area=1950,
        template_text="ABC",
    )
    enabled = PixelDiffConfig(
        sensitive_recall_similarity_filter_enabled=True,
        sensitive_recall_similarity_search_radius=16,
        sensitive_recall_similarity_iou_threshold=0.62,
    )
    disabled = PixelDiffConfig(sensitive_recall_similarity_filter_enabled=False)

    kept = text_layer.filter_recalled_similarity_regions(
        [region], scan_binary=scan, template_binary=template, config=enabled
    )
    untouched = text_layer.filter_recalled_similarity_regions(
        [region], scan_binary=scan, template_binary=template, config=disabled
    )

    assert kept == [region]
    assert untouched == [region]


def test_sensitive_spans_do_not_expand_into_adjacent_chinese() -> None:
    spans = text_layer._sensitive_spans("住址B座9层，first、second项")

    assert [(value, start, end) for value, start, end in spans] == [
        ("B", 2, 3),
        ("9", 4, 5),
        ("first", 7, 12),
        ("second", 13, 19),
    ]


def test_pdf_rows_are_cached_for_unchanged_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fitz

    template = _write_template(tmp_path / "cached.pdf")
    real_open = fitz.open
    calls = 0

    def counting_open(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(fitz, "open", counting_open)

    first = text_layer._extract_pdf_rows(template, page=0, dpi=300)
    second = text_layer._extract_pdf_rows(template, page=0, dpi=300)

    assert first == second
    assert calls == 1


def test_pdf_image_keep_mask_ignores_embedded_document_image(tmp_path: Path) -> None:
    pdf = _write_pdf_with_image(tmp_path / "with_image.pdf", full_page=False)
    config = PixelDiffConfig(pdf_image_region_filter_enabled=True)

    mask = text_layer.build_pdf_image_keep_mask(
        pdf,
        pdf,
        page=0,
        image_shape=(400, 600),
        config=config,
    )

    assert mask is not None
    assert mask[200, 300] == 0
    assert mask[30, 30] == 255


def test_pdf_image_keep_mask_does_not_hide_near_full_page_raster(tmp_path: Path) -> None:
    pdf = _write_pdf_with_image(tmp_path / "full_page.pdf", full_page=True)
    config = PixelDiffConfig(
        pdf_image_region_filter_enabled=True,
        pdf_image_region_max_page_ratio=0.80,
    )

    mask = text_layer.build_pdf_image_keep_mask(
        pdf,
        pdf,
        page=0,
        image_shape=(400, 600),
        config=config,
    )

    assert mask is None or np.all(mask == 255)


def _write_template(path: Path) -> Path:
    import fitz

    document = fitz.open()
    page = document.new_page(width=216, height=120)
    page.insert_text((20, 50), "phone: 010-6528")
    page.insert_text((20, 80), "first, second")
    document.save(path)
    document.close()
    return path


def _write_pdf_with_image(path: Path, *, full_page: bool) -> Path:
    import cv2
    import fitz

    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((15, 20), "document text")
    image = np.full((40, 60, 3), 120, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    rect = fitz.Rect(5, 5, 295, 195) if full_page else fitz.Rect(100, 70, 200, 140)
    page.insert_image(rect, stream=encoded.tobytes())
    document.save(path)
    document.close()
    return path


def test_build_text_anchor_line_protects_inserted_text() -> None:
    template = text_layer.TextRow(
        "北京元甲律师事务所",
        tuple(
            text_layer.TextChar(ch, i * 10, 10, i * 10 + 8, 24)
            for i, ch in enumerate("北京元甲律师事务所")
        ),
    )
    scan_text = "北京市元甲律师事务所"
    scan = text_layer.TextRow(
        scan_text,
        tuple(
            text_layer.TextChar(ch, i * 10, 10, i * 10 + 8, 24) for i, ch in enumerate(scan_text)
        ),
    )

    line = text_layer._build_text_anchor_line(scan, template, min_equal_chars=2, padding=3)

    assert line is not None
    assert line.anchors[0][1] - line.anchors[0][0] == 0
    assert line.anchors[-1][1] - line.anchors[-1][0] == 10
    assert line.protected_intervals
