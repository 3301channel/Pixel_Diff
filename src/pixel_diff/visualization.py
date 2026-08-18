"""Visualization helpers for region boxes and text ghost comparison images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pixel_diff.exceptions import DimensionMismatchError
from pixel_diff.models import DifferenceRegion

_REGION_COLOR = (0, 0, 255)
_TEXT_COLOR = (0, 0, 255)
_LINE_THICKNESS = 2
_FONT_SCALE = 0.6
_CLASS_COLORS = {
    "added": (0, 180, 0),
    "deleted": (0, 0, 255),
    "modified": (0, 165, 255),
    "displaced": (255, 0, 0),
}

GHOST_BACKGROUND_BGR = (205, 154, 154)
GHOST_TEMPLATE_ONLY_BGR = (0, 0, 255)
GHOST_SCAN_ONLY_BGR = (255, 255, 0)
GHOST_OVERLAP_BGR = (245, 245, 245)


def draw_regions(
    image_bgr: np.ndarray,
    regions: list[DifferenceRegion],
    show_classification_labels: bool = False,
) -> np.ndarray:
    """Draw red boxes and numeric labels for suspected difference regions."""

    output = image_bgr.copy()
    for region in regions:
        color = (
            _CLASS_COLORS.get(region.change_type or "", _REGION_COLOR)
            if show_classification_labels
            else _REGION_COLOR
        )
        cv2.rectangle(
            output,
            (region.x, region.y),
            (region.x + region.width, region.y + region.height),
            color,
            _LINE_THICKNESS,
        )
        if not show_classification_labels:
            cv2.putText(
                output,
                str(region.id),
                (region.x + 4, region.y + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                _FONT_SCALE,
                _TEXT_COLOR,
                2,
            )
    if show_classification_labels:
        output = _draw_classification_text(output, regions)
    return output


def _region_label(region: DifferenceRegion, show_classification_labels: bool) -> str:
    if show_classification_labels and region.change_label:
        return f"{region.id} {region.change_label}"
    return str(region.id)


def _draw_classification_text(
    image_bgr: np.ndarray, regions: list[DifferenceRegion]
) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _load_label_font()
    for region in regions:
        bgr = _CLASS_COLORS.get(region.change_type or "", _REGION_COLOR)
        rgb_color = (bgr[2], bgr[1], bgr[0])
        y = region.y - 22 if region.y >= 24 else region.y + 3
        draw.text(
            (max(0, region.x + 3), max(0, y)),
            _region_label(region, True),
            font=font,
            fill=rgb_color,
            stroke_width=1,
            stroke_fill=(255, 255, 255),
        )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _load_label_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), 18)
    return ImageFont.load_default()


def draw_text_ghost_comparison(
    template_binary: np.ndarray,
    scan_binary: np.ndarray,
    match_tolerance: int = 0,
) -> np.ndarray:
    """Create a BGR ghost image from two binary images.

    Binary convention: text foreground is 0 and background is 255.
    """

    if template_binary.ndim != 2 or scan_binary.ndim != 2:
        raise DimensionMismatchError("visualization: ghost inputs must be 2D binary images")
    if template_binary.shape != scan_binary.shape:
        raise DimensionMismatchError("visualization: ghost inputs must have identical shapes")

    height, width = template_binary.shape
    ghost = np.full((height, width, 3), GHOST_BACKGROUND_BGR, dtype=np.uint8)

    template_fg = template_binary == 0
    template_bg = template_binary == 255
    scan_fg = scan_binary == 0
    scan_bg = scan_binary == 255

    if match_tolerance > 0:
        distance_to_scan = cv2.distanceTransform(
            np.where(scan_fg, 0, 255).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        distance_to_template = cv2.distanceTransform(
            np.where(template_fg, 0, 255).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        template_matched = template_fg & (distance_to_scan <= match_tolerance)
        scan_matched = scan_fg & (distance_to_template <= match_tolerance)
        template_only = template_fg & ~template_matched
        scan_only = scan_fg & ~scan_matched
        ghost[template_only] = GHOST_TEMPLATE_ONLY_BGR
        ghost[scan_only] = GHOST_SCAN_ONLY_BGR
        ghost[template_matched | scan_matched] = GHOST_OVERLAP_BGR
    else:
        template_only = template_fg & scan_bg
        scan_only = template_bg & scan_fg
        ghost[template_only] = GHOST_TEMPLATE_ONLY_BGR
        ghost[scan_only] = GHOST_SCAN_ONLY_BGR
        ghost[template_fg & scan_fg] = GHOST_OVERLAP_BGR
    return ghost
