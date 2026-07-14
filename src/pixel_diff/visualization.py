"""Visualization helpers for region boxes and text ghost comparison images."""

from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.exceptions import DimensionMismatchError
from pixel_diff.models import DifferenceRegion

_REGION_COLOR = (0, 0, 255)
_TEXT_COLOR = (0, 0, 255)
_LINE_THICKNESS = 2
_FONT_SCALE = 0.6

GHOST_BACKGROUND_BGR = (205, 154, 154)
GHOST_TEMPLATE_ONLY_BGR = (0, 0, 255)
GHOST_SCAN_ONLY_BGR = (255, 255, 0)
GHOST_OVERLAP_BGR = (245, 245, 245)


def draw_regions(
    image_bgr: np.ndarray,
    regions: list[DifferenceRegion],
) -> np.ndarray:
    """Draw red boxes and numeric labels for suspected difference regions."""

    output = image_bgr.copy()
    for region in regions:
        cv2.rectangle(
            output,
            (region.x, region.y),
            (region.x + region.width, region.y + region.height),
            _REGION_COLOR,
            _LINE_THICKNESS,
        )
        cv2.putText(
            output,
            str(region.id),
            (region.x + 4, region.y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            _FONT_SCALE,
            _TEXT_COLOR,
            2,
        )
    return output


def draw_text_ghost_comparison(
    template_binary: np.ndarray,
    scan_binary: np.ndarray,
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

    ghost[template_fg & scan_bg] = GHOST_TEMPLATE_ONLY_BGR
    ghost[template_bg & scan_fg] = GHOST_SCAN_ONLY_BGR
    ghost[template_fg & scan_fg] = GHOST_OVERLAP_BGR
    return ghost
