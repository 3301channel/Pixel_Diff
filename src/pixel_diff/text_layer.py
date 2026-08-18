"""PDF text-layer helpers used as a conservative signal for digital PDFs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from pixel_diff.models import DifferenceRegion, PixelDiffConfig
from pixel_diff.region_utils import renumber_regions
from pixel_diff.regions import _best_local_iou_vectorized


@dataclass(frozen=True)
class TextChar:
    """One rendered PDF character in image-pixel coordinates."""

    value: str
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class TextRow:
    """A visual row assembled from PDF raw text spans."""

    text: str
    chars: tuple[TextChar, ...]

    @property
    def center_y(self) -> float:
        return sum((char.y0 + char.y1) / 2 for char in self.chars) / len(self.chars)


@dataclass(frozen=True)
class TextAnchorLine:
    scan_center_y: int
    template_center_y: int
    anchors: tuple[tuple[float, float], ...]
    protected_intervals: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PdfImageLayout:
    page_width: float
    page_height: float
    has_text: bool
    rectangles: tuple[tuple[float, float, float, float], ...]


def build_pdf_image_keep_mask(
    scan_path: str | Path,
    template_path: str | Path,
    page: int,
    image_shape: tuple[int, int],
    config: PixelDiffConfig,
) -> np.ndarray | None:
    """Mask embedded document images while preserving full-page scanned pages."""
    if not config.pdf_image_region_filter_enabled:
        return None

    height, width = image_shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    painted = False
    for path in (scan_path, template_path):
        source = Path(path)
        if source.suffix.lower() != ".pdf" or not source.exists():
            continue
        stat = source.stat()
        layout = _extract_pdf_image_layout(
            str(source.resolve()), stat.st_mtime_ns, stat.st_size, page
        )
        if layout is None or not layout.has_text:
            continue
        page_area = layout.page_width * layout.page_height
        for x0, y0, x1, y1 in layout.rectangles:
            image_ratio = ((x1 - x0) * (y1 - y0)) / page_area if page_area > 0 else 1.0
            if image_ratio > config.pdf_image_region_max_page_ratio:
                continue
            px0 = int(np.floor(x0 * width / layout.page_width)) - config.pdf_image_region_padding
            py0 = int(np.floor(y0 * height / layout.page_height)) - config.pdf_image_region_padding
            px1 = int(np.ceil(x1 * width / layout.page_width)) + config.pdf_image_region_padding
            py1 = int(np.ceil(y1 * height / layout.page_height)) + config.pdf_image_region_padding
            mask[max(0, py0) : min(height, py1), max(0, px0) : min(width, px1)] = 0
            painted = True
    return mask if painted else None


@lru_cache(maxsize=64)
def _extract_pdf_image_layout(
    resolved_path: str,
    _mtime_ns: int,
    _size: int,
    page_number: int,
) -> PdfImageLayout | None:
    try:
        import fitz

        with fitz.open(resolved_path) as document:
            if not 0 <= page_number < document.page_count:
                return None
            pdf_page = document.load_page(page_number)
            rectangles: list[tuple[float, float, float, float]] = []
            for image in pdf_page.get_images(full=True):
                for rect in pdf_page.get_image_rects(image[0]):
                    rectangles.append((rect.x0, rect.y0, rect.x1, rect.y1))
            return PdfImageLayout(
                page_width=float(pdf_page.rect.width),
                page_height=float(pdf_page.rect.height),
                has_text=bool(pdf_page.get_text("text").strip()),
                rectangles=tuple(rectangles),
            )
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def build_unchanged_text_mask(
    scan_path: str | Path,
    template_path: str | Path,
    page: int,
    image_shape: tuple[int, int],
    config: PixelDiffConfig,
) -> np.ndarray | None:
    """Return a keep-mask that removes stable, unchanged PDF characters.

    The mask is only produced when both inputs are PDFs with extractable text. A zero
    pixel in the returned mask means "ignore differences here".
    """

    if not config.pdf_text_layer_filter:
        return None

    row_pairs = _paired_rows(scan_path, template_path, page, config.dpi)
    if not row_pairs:
        return None

    height, width = image_shape
    mask = np.full((height, width), 255, dtype=np.uint8)

    for scan_row, template_row in row_pairs:
        matcher = SequenceMatcher(a=template_row.text, b=scan_row.text, autojunk=False)
        for tag, template_start, template_end, scan_start, _scan_end in matcher.get_opcodes():
            if tag != "equal":
                continue
            for offset in range(template_end - template_start):
                template_char = template_row.chars[template_start + offset]
                scan_char = scan_row.chars[scan_start + offset]
                if template_char.value != scan_char.value:
                    continue
                if not _same_position(template_char, scan_char, config.pdf_text_position_tolerance):
                    continue
                _paint_char(mask, template_char, config.pdf_text_mask_padding, 0)

    return mask


def extract_text_anchor_lines(
    scan_path: str | Path,
    template_path: str | Path,
    page: int,
    config: PixelDiffConfig,
) -> list[TextAnchorLine]:
    if not config.pdf_text_anchor_alignment_enabled:
        return []
    return [
        line
        for scan_row, template_row in _paired_rows(scan_path, template_path, page, config.dpi)
        if (line := _build_text_anchor_line(
            scan_row,
            template_row,
            config.pdf_text_anchor_min_equal_chars,
            config.pdf_text_anchor_protection_padding,
        )) is not None
    ]


def has_pdf_text_layer(path: str | Path, page: int, dpi: int) -> bool:
    """Return whether a PDF page exposes at least one extractable text row."""
    return bool(_extract_pdf_rows(path, page, dpi))


def _build_text_anchor_line(
    scan_row: TextRow,
    template_row: TextRow,
    min_equal_chars: int,
    padding: int,
) -> TextAnchorLine | None:
    matcher = SequenceMatcher(a=template_row.text, b=scan_row.text, autojunk=False)
    anchors: list[tuple[float, float]] = []
    protected: list[tuple[int, int]] = []
    for tag, t0, t1, s0, s1 in matcher.get_opcodes():
        if tag == "equal" and t1 - t0 >= min_equal_chars:
            for template_char, scan_char in zip(
                template_row.chars[t0:t1], scan_row.chars[s0:s1], strict=True
            ):
                anchors.append(
                    ((template_char.x0 + template_char.x1) / 2, (scan_char.x0 + scan_char.x1) / 2)
                )
            continue
        template_chars = template_row.chars[t0:t1]
        scan_chars = scan_row.chars[s0:s1]
        if template_chars:
            x0 = min(char.x0 for char in template_chars)
            x1 = max(char.x1 for char in template_chars)
        else:
            boundary = (
                template_row.chars[t0].x0
                if t0 < len(template_row.chars)
                else template_row.chars[-1].x1
            )
            inserted_width = sum(max(1, char.x1 - char.x0) for char in scan_chars)
            x0, x1 = boundary - inserted_width // 2, boundary + inserted_width // 2
        protected.append((max(0, x0 - padding), x1 + padding))
    if len(anchors) < 2 or not protected:
        return None
    return TextAnchorLine(
        scan_center_y=round(scan_row.center_y),
        template_center_y=round(template_row.center_y),
        anchors=tuple(anchors),
        protected_intervals=tuple(protected),
    )


def extract_text_difference_regions(
    scan_path: str | Path,
    template_path: str | Path,
    page: int,
    image_shape: tuple[int, int],
    config: PixelDiffConfig,
) -> list[DifferenceRegion]:
    """Create supplemental regions for PDF text-layer character differences."""

    if not config.pdf_text_layer_filter:
        return []

    row_pairs = _paired_rows(scan_path, template_path, page, config.dpi)
    if not row_pairs:
        return []

    height, width = image_shape
    regions: list[DifferenceRegion] = []
    for scan_row, template_row in row_pairs:
        matcher = SequenceMatcher(a=template_row.text, b=scan_row.text, autojunk=False)
        for tag, template_start, template_end, scan_start, scan_end in matcher.get_opcodes():
            if tag == "equal":
                continue
            changed_chars = (
                list(template_row.chars[template_start:template_end])
                + list(scan_row.chars[scan_start:scan_end])
            )
            region = _region_from_chars(
                changed_chars,
                width=width,
                height=height,
                padding=config.pdf_text_region_padding,
                text_layer_protected=True,
            )
            if region is not None:
                regions.append(region)

    return renumber_regions(regions)


def extract_sensitive_text_recall_regions(
    template_path: str | Path,
    page: int,
    diff_mask: np.ndarray,
    config: PixelDiffConfig,
) -> list[DifferenceRegion]:
    """Create high-recall regions for fragmented differences over digit/Latin runs."""
    if not config.sensitive_text_recall_enabled:
        return []

    height, width = diff_mask.shape[:2]
    regions: list[DifferenceRegion] = []
    for row in _extract_pdf_rows(template_path, page, config.dpi):
        for token, start, end in _sensitive_spans(row.text):
            chars = list(row.chars[start:end])
            region = _region_from_chars(
                chars,
                width=width,
                height=height,
                padding=config.sensitive_text_recall_padding,
            )
            if region is None:
                continue
            roi = diff_mask[
                region.y : region.y + region.height,
                region.x : region.x + region.width,
            ]
            if roi.size == 0:
                continue
            density = float(np.count_nonzero(roi)) / float(roi.size)
            if density < config.sensitive_text_recall_min_density:
                continue
            regions.append(
                DifferenceRegion(
                    id=0,
                    x=region.x,
                    y=region.y,
                    width=region.width,
                    height=region.height,
                    area=float(np.count_nonzero(roi)),
                    template_text=token,
                )
            )
    return renumber_regions(regions)


def filter_recalled_similarity_regions(
    regions: list[DifferenceRegion],
    *,
    scan_binary: np.ndarray,
    template_binary: np.ndarray,
    config: PixelDiffConfig,
    template_text_by_region: dict[int, str] | None = None,
) -> list[DifferenceRegion]:
    """Remove recalled text regions explained by a bounded rigid translation."""
    if (
        not config.sensitive_recall_similarity_filter_enabled
        or not regions
        or scan_binary.shape != template_binary.shape
    ):
        return regions

    scan_foreground = (scan_binary == 0).astype(np.uint8)
    template_foreground = (template_binary == 0).astype(np.uint8)
    kept: list[DifferenceRegion] = []
    for region in regions:
        search_radius = config.sensitive_recall_similarity_search_radius
        template_text = region.template_text or (
            template_text_by_region or {}
        ).get(region.id)
        if _is_page_number_recall_candidate(
            region,
            template_binary.shape,
            config,
            template_text=template_text,
        ):
            search_radius = config.sensitive_recall_page_number_search_radius
        best_iou = _best_local_iou_vectorized(
            region,
            scan_foreground,
            template_foreground,
            padding=config.local_similarity_padding,
            search_radius=search_radius,
        )
        if best_iou < config.sensitive_recall_similarity_iou_threshold:
            kept.append(region)
    return renumber_regions(kept)


def _is_page_number_recall_candidate(
    region: DifferenceRegion,
    image_shape: tuple[int, ...],
    config: PixelDiffConfig,
    *,
    template_text: str | None,
) -> bool:
    text = (template_text or "").strip()
    if not text.isdigit():
        return False
    height, width = image_shape[:2]
    center_x = region.x + region.width / 2.0
    return (
        region.y >= height * config.risk_review_page_number_bottom_ratio
        and abs(center_x - width / 2.0)
        <= width * config.risk_review_page_number_center_tolerance_ratio
        and region.width <= config.risk_review_page_number_max_width
        and region.height <= config.risk_review_page_number_max_height
    )


def _sensitive_spans(text: str) -> list[tuple[str, int, int]]:
    """Return bounded Latin words and numeric expressions from mixed-language text."""
    return [
        (match.group(), match.start(), match.end())
        for match in re.finditer(r"[A-Za-z]+|[0-9]+(?:[-./][0-9]+)*", text)
    ]


def compare_text_regions(
    regions: list[DifferenceRegion],
    scan_path: str | Path,
    template_path: str | Path,
    page: int,
    dpi: int,
) -> list[dict[str, object]]:
    """Return PDF text-layer annotations for regions without filtering them."""

    row_pairs = _paired_rows(scan_path, template_path, page, dpi)
    if not row_pairs:
        return []

    annotations: list[dict[str, object]] = []
    for region in regions:
        template_text = "".join(
            char.value
            for _scan_row, template_row in row_pairs
            for char in template_row.chars
            if _char_intersects_region(char, region)
        )
        scan_text = "".join(
            char.value
            for scan_row, _template_row in row_pairs
            for char in scan_row.chars
            if _char_intersects_region(char, region)
        )
        if not template_text and not scan_text:
            continue
        annotations.append(
            {
                "region_id": region.id,
                "bbox": [region.x, region.y, region.width, region.height],
                "template_text": template_text,
                "scan_text": scan_text,
                "is_identical": template_text == scan_text,
            }
        )
    return annotations


def extract_template_text_by_region(
    regions: list[DifferenceRegion],
    template_path: str | Path,
    page: int,
    dpi: int,
    padding: int,
) -> dict[int, str]:
    """Return template PDF text intersecting each region."""

    template_rows = _extract_pdf_rows(template_path, page, dpi)
    if not template_rows:
        return {}

    template_text_by_region: dict[int, str] = {}
    for region in regions:
        expanded = DifferenceRegion(
            id=region.id,
            x=region.x - padding,
            y=region.y - padding,
            width=region.width + padding * 2,
            height=region.height + padding * 2,
            area=region.area,
        )
        text = "".join(
            char.value
            for row in template_rows
            for char in row.chars
            if _char_intersects_region(char, expanded)
        )
        if text:
            template_text_by_region[region.id] = text
    return template_text_by_region


def merge_regions(
    pixel_regions: list[DifferenceRegion],
    supplemental_regions: list[DifferenceRegion],
    config: PixelDiffConfig | None = None,
) -> list[DifferenceRegion]:
    """Merge supplemental text regions with pixel regions and renumber.

    A supplemental text region (e.g. an appended/inserted character like ``_P``
    after ``何平``) is only merged into a pixel region when the two genuinely
    overlap. The allowed gap is controlled by ``merge_region_expand_padding``
    (default 0 = require a real intersection). This prevents appended characters
    from being silently absorbed into the parent word's bounding box, which made
    them vanish as separate detected regions.
    """

    expand = config.merge_region_expand_padding if config is not None else 0
    merged = list(pixel_regions)
    for supplemental in supplemental_regions:
        merged_index = _find_merge_target(merged, supplemental, padding=expand)
        if merged_index is None:
            merged.append(supplemental)
        else:
            merged[merged_index] = _union_region(merged[merged_index], supplemental)

    merged.sort(key=lambda region: (region.y, region.x, -region.area))
    return renumber_regions(merged)


def _paired_rows(
    scan_path: str | Path,
    template_path: str | Path,
    page: int,
    dpi: int,
) -> list[tuple[TextRow, TextRow]]:
    scan_rows = _extract_pdf_rows(scan_path, page, dpi)
    template_rows = _extract_pdf_rows(template_path, page, dpi)
    if not scan_rows or not template_rows:
        return []

    pairs: list[tuple[TextRow, TextRow]] = []
    used_scan: set[int] = set()
    max_y_distance = max(16.0, dpi / 12.0)

    for template_row in template_rows:
        best_index: int | None = None
        best_distance = max_y_distance
        for index, scan_row in enumerate(scan_rows):
            if index in used_scan:
                continue
            distance = abs(scan_row.center_y - template_row.center_y)
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None:
            used_scan.add(best_index)
            pairs.append((scan_rows[best_index], template_row))

    return pairs


def _extract_pdf_rows(path: str | Path, page: int, dpi: int) -> list[TextRow]:
    file = Path(path)
    if file.suffix.lower() != ".pdf":
        return []
    try:
        resolved = file.resolve()
        stat = resolved.stat()
    except OSError:
        return []
    return list(
        _extract_pdf_rows_cached(
            str(resolved),
            page,
            dpi,
            stat.st_mtime_ns,
            stat.st_size,
        )
    )


@lru_cache(maxsize=128)
def _extract_pdf_rows_cached(
    path: str,
    page: int,
    dpi: int,
    _mtime_ns: int,
    _size: int,
) -> tuple[TextRow, ...]:
    """Parse immutable text rows once for an unchanged PDF page."""
    file = Path(path)

    try:
        import fitz

        document = fitz.open(file)
        if page < 0 or page >= document.page_count:
            document.close()
            return ()
        raw = document[page].get_text("rawdict")
        document.close()
    except Exception:
        return ()

    scale = dpi / 72.0
    rows: list[list[TextChar]] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_chars: list[TextChar] = []
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    bbox = char.get("bbox")
                    value = char.get("c", "")
                    if not bbox or not value:
                        continue
                    x0, y0, x1, y1 = bbox
                    line_chars.append(
                        TextChar(
                            value=value,
                            x0=int(round(float(x0) * scale)),
                            y0=int(round(float(y0) * scale)),
                            x1=int(round(float(x1) * scale)),
                            y1=int(round(float(y1) * scale)),
                        )
                    )
            if line_chars:
                _append_to_row(rows, line_chars)

    result = []
    for chars in rows:
        ordered = tuple(sorted(chars, key=lambda char: (char.x0, char.y0)))
        result.append(TextRow(text="".join(char.value for char in ordered), chars=ordered))
    result.sort(key=lambda row: (row.center_y, row.chars[0].x0))
    return tuple(result)


def _append_to_row(rows: list[list[TextChar]], chars: list[TextChar]) -> None:
    center_y = sum((char.y0 + char.y1) / 2 for char in chars) / len(chars)
    for row in rows:
        row_center_y = sum((char.y0 + char.y1) / 2 for char in row) / len(row)
        if abs(row_center_y - center_y) <= 12:
            row.extend(chars)
            return
    rows.append(chars)


def _same_position(template_char: TextChar, scan_char: TextChar, tolerance: int) -> bool:
    return max(
        abs(template_char.x0 - scan_char.x0),
        abs(template_char.y0 - scan_char.y0),
        abs(template_char.x1 - scan_char.x1),
        abs(template_char.y1 - scan_char.y1),
    ) <= tolerance


def _paint_char(mask: np.ndarray, char: TextChar, padding: int, value: int) -> None:
    height, width = mask.shape[:2]
    cv2.rectangle(
        mask,
        (max(0, char.x0 - padding), max(0, char.y0 - padding)),
        (min(width - 1, char.x1 + padding), min(height - 1, char.y1 + padding)),
        value,
        -1,
    )


def _char_intersects_region(char: TextChar, region: DifferenceRegion) -> bool:
    return not (
        char.x1 < region.x
        or char.x0 > region.x + region.width
        or char.y1 < region.y
        or char.y0 > region.y + region.height
    )


def _region_from_chars(
    chars: list[TextChar],
    width: int,
    height: int,
    padding: int,
    text_layer_protected: bool = False,
) -> DifferenceRegion | None:
    if not chars:
        return None
    x0 = max(0, min(char.x0 for char in chars) - padding)
    y0 = max(0, min(char.y0 for char in chars) - padding)
    x1 = min(width, max(char.x1 for char in chars) + padding)
    y1 = min(height, max(char.y1 for char in chars) + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    return DifferenceRegion(
        id=0,
        x=x0,
        y=y0,
        width=x1 - x0,
        height=y1 - y0,
        area=float((x1 - x0) * (y1 - y0)),
        text_layer_protected=text_layer_protected,
    )


def _find_merge_target(
    regions: list[DifferenceRegion],
    candidate: DifferenceRegion,
    padding: int = 0,
) -> int | None:
    for index, region in enumerate(regions):
        if _intersection_area(region, candidate) > 0:
            return index
        if padding > 0 and _expanded_intersection_area(region, candidate, padding=padding) > 0:
            return index
    return None


def _intersection_area(left: DifferenceRegion, right: DifferenceRegion) -> int:
    x0 = max(left.x, right.x)
    y0 = max(left.y, right.y)
    x1 = min(left.x + left.width, right.x + right.width)
    y1 = min(left.y + left.height, right.y + right.height)
    return max(0, x1 - x0) * max(0, y1 - y0)


def _expanded_intersection_area(
    left: DifferenceRegion,
    right: DifferenceRegion,
    padding: int,
) -> int:
    expanded = DifferenceRegion(
        id=left.id,
        x=left.x - padding,
        y=left.y - padding,
        width=left.width + padding * 2,
        height=left.height + padding * 2,
        area=left.area,
    )
    return _intersection_area(expanded, right)


def _union_region(left: DifferenceRegion, right: DifferenceRegion) -> DifferenceRegion:
    x0 = min(left.x, right.x)
    y0 = min(left.y, right.y)
    x1 = max(left.x + left.width, right.x + right.width)
    y1 = max(left.y + left.height, right.y + right.height)
    return DifferenceRegion(
        id=left.id,
        x=x0,
        y=y0,
        width=x1 - x0,
        height=y1 - y0,
        area=max(left.area, right.area, float((x1 - x0) * (y1 - y0))),
        template_text=_join_optional_text(left.template_text, right.template_text),
        text_layer_protected=left.text_layer_protected or right.text_layer_protected,
    )


def _join_optional_text(left: str | None, right: str | None) -> str | None:
    if left and right:
        return left if right in left else f"{left}{right}"
    return left or right
