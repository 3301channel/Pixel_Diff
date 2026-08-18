"""PDF text-layer anchored horizontal correction before visual registration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pixel_diff.models import PixelDiffConfig
from pixel_diff.text_layer import (
    TextAnchorLine,
    TextChar,
    TextRow,
    _build_text_anchor_line,
    _extract_pdf_rows,
)


@dataclass(frozen=True)
class TextAnchorAlignmentResult:
    aligned_bgr: np.ndarray
    checked_lines: int
    applied_lines: int
    anchors: int
    protected_intervals: int
    before_iou: float
    after_iou: float


def extract_ocr_text_anchor_lines(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    template_path: str,
    page: int,
    config: PixelDiffConfig,
) -> list[TextAnchorLine]:
    if not config.pdf_text_anchor_ocr_fallback_enabled:
        return []
    if _page_foreground_iou(scan_bgr, template_bgr) > config.pdf_text_anchor_ocr_max_page_iou:
        return []
    template_rows = _extract_pdf_rows(template_path, page, config.dpi)
    if not template_rows:
        return []
    scan_rows = _ocr_rows(scan_bgr, config)
    if not scan_rows:
        return []
    lines: list[TextAnchorLine] = []
    used: set[int] = set()
    for template_row in template_rows:
        candidates = [
            (
                abs(
                    row.center_y / scan_bgr.shape[0] - template_row.center_y / template_bgr.shape[0]
                ),
                index,
                row,
            )
            for index, row in enumerate(scan_rows)
            if index not in used
        ]
        if not candidates:
            continue
        distance, index, scan_row = min(candidates)
        if distance > 0.025:
            continue
        line = _build_text_anchor_line(
            scan_row,
            template_row,
            config.pdf_text_anchor_min_equal_chars,
            config.pdf_text_anchor_protection_padding,
        )
        if line is not None:
            used.add(index)
            lines.append(line)
    return lines


def _ocr_rows(scan_bgr: np.ndarray, config: PixelDiffConfig) -> list[TextRow]:
    try:
        from pixel_diff.risk_review import _rapidocr_engine

        engine = _rapidocr_engine()
        if engine is None:
            return []
        scale = config.pdf_text_anchor_ocr_scale
        image = cv2.resize(scan_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        raw = engine(image)
        entries = raw[0] if isinstance(raw, tuple) and raw else raw
    except Exception:
        return []
    if not isinstance(entries, (list, tuple)):
        return []
    rows: list[TextRow] = []
    for entry in entries or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        box, text, confidence = entry[:3]
        if not text or float(confidence) < config.pdf_text_anchor_ocr_min_confidence:
            continue
        xs = [float(point[0]) / scale for point in box]
        ys = [float(point[1]) / scale for point in box]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        char_width = (x1 - x0) / len(str(text))
        chars = tuple(
            TextChar(
                value=char,
                x0=round(x0 + index * char_width),
                y0=round(y0),
                x1=round(x0 + (index + 1) * char_width),
                y1=round(y1),
            )
            for index, char in enumerate(str(text))
        )
        rows.append(TextRow(str(text), chars))
    return sorted(rows, key=lambda row: row.center_y)


def align_by_text_anchors_bgr(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    lines: list[TextAnchorLine],
    config: PixelDiffConfig,
) -> TextAnchorAlignmentResult:
    if not lines:
        return TextAnchorAlignmentResult(scan_bgr, 0, 0, 0, 0, 0.0, 0.0)
    output = scan_bgr.copy()
    scan_fg, template_fg = _foreground(scan_bgr), _foreground(template_bgr)
    height, width = scan_fg.shape
    applied = anchors = protected_count = 0
    before_scores: list[float] = []
    after_scores: list[float] = []
    for line in lines:
        half = config.line_affine_band_half_height
        sy0, sy1 = max(0, line.scan_center_y - half), min(height, line.scan_center_y + half + 1)
        ty0, ty1 = (
            max(0, line.template_center_y - half),
            min(height, line.template_center_y + half + 1),
        )
        band_height = min(sy1 - sy0, ty1 - ty0)
        if band_height <= 0:
            continue
        offsets = _offset_map(line, width, config)
        if offsets is None:
            continue
        corrected = _remap(output[sy0 : sy0 + band_height], offsets)
        corrected_fg = _foreground(corrected)
        common_width = min(corrected_fg.shape[1], template_fg.shape[1])
        template_band = template_fg[ty0 : ty0 + band_height, :common_width]
        before = _iou(scan_fg[sy0 : sy0 + band_height, :common_width], template_band)
        after = _iou(corrected_fg[:, :common_width], template_band)
        if after - before < config.pdf_text_anchor_min_improvement:
            continue
        for x0, x1 in line.protected_intervals:
            corrected[:, x0:x1] = output[sy0 : sy0 + band_height, x0:x1]
        output[sy0 : sy0 + band_height] = corrected
        applied += 1
        anchors += len(line.anchors)
        protected_count += len(line.protected_intervals)
        before_scores.append(before)
        after_scores.append(after)
    mean_before = float(np.mean(before_scores)) if before_scores else 0.0
    mean_after = float(np.mean(after_scores)) if after_scores else 0.0
    if (
        config.pdf_text_anchor_ocr_fallback_enabled
        and mean_before > config.pdf_text_anchor_ocr_max_before_iou
    ):
        return TextAnchorAlignmentResult(scan_bgr, len(lines), 0, 0, 0, 0.0, 0.0)
    return TextAnchorAlignmentResult(
        output,
        len(lines),
        applied,
        anchors,
        protected_count,
        mean_before,
        mean_after,
    )


def _offset_map(line: TextAnchorLine, width: int, config: PixelDiffConfig) -> np.ndarray | None:
    template_x = np.array([item[0] for item in line.anchors], dtype=np.float32)
    offsets_at_anchor = np.array([item[1] - item[0] for item in line.anchors], dtype=np.float32)
    median_offset = float(np.median(offsets_at_anchor))
    if abs(median_offset) > 400:
        return None
    if np.max(np.abs(offsets_at_anchor - median_offset)) > config.pdf_text_anchor_max_shift:
        return None
    if np.any(np.diff(template_x) <= 0):
        return None
    offsets = np.interp(np.arange(width), template_x, offsets_at_anchor).astype(np.float32)
    for x0, x1 in line.protected_intervals:
        left = max(0, min(len(template_x) - 1, int(np.searchsorted(template_x, x0) - 1)))
        right = min(len(template_x) - 1, int(np.searchsorted(template_x, x1)))
        if left == right:
            continue
        span = max(1.0, template_x[right] - template_x[left])
        scale_delta = abs(float(offsets_at_anchor[right] - offsets_at_anchor[left])) / span
        if scale_delta > config.pdf_text_anchor_max_scale_delta:
            midpoint = (x0 + x1) // 2
            offsets[max(0, x0) : midpoint] = offsets_at_anchor[left]
            offsets[midpoint : min(width, x1)] = offsets_at_anchor[right]
    return offsets


def _remap(image: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(
        np.arange(w, dtype=np.float32) + offsets, np.arange(h, dtype=np.float32)
    )
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _foreground(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary.astype(bool)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.count_nonzero(left | right))
    return 1.0 if union == 0 else int(np.count_nonzero(left & right)) / union


def _page_foreground_iou(scan_bgr: np.ndarray, template_bgr: np.ndarray) -> float:
    scan = _foreground(scan_bgr).astype(np.uint8)
    template = _foreground(template_bgr)
    scan = cv2.resize(
        scan,
        (template.shape[1], template.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    return _iou(scan, template)
