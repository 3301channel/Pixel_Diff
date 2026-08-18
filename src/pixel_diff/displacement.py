"""Pair opposite directional residuals that represent moved content."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from pixel_diff.models import DifferenceRegion, PixelDiffConfig
from pixel_diff.region_utils import renumber_regions


def pair_displaced_regions(
    regions: list[DifferenceRegion],
    scan_binary: np.ndarray,
    template_binary: np.ndarray,
    config: PixelDiffConfig,
) -> tuple[list[DifferenceRegion], int]:
    """Merge mutual-best matching additions and deletions into union regions."""
    if not config.displacement_pairing_enabled or len(regions) < 2:
        return regions, 0

    scan_foreground = scan_binary == 0
    template_foreground = template_binary == 0
    added = scan_foreground & ~template_foreground
    deleted = template_foreground & ~scan_foreground
    additions: list[tuple[int, np.ndarray]] = []
    deletions: list[tuple[int, np.ndarray]] = []
    for index, region in enumerate(regions):
        added_pixels = _crop(added, region, config.displacement_pairing_padding)
        deleted_pixels = _crop(deleted, region, config.displacement_pairing_padding)
        added_count = int(np.count_nonzero(added_pixels))
        deleted_count = int(np.count_nonzero(deleted_pixels))
        directional_total = added_count + deleted_count
        if directional_total == 0:
            continue
        if added_count / directional_total >= config.displacement_pairing_min_direction_ratio:
            additions.append((index, _normalize_shape(added_pixels)))
        elif deleted_count / directional_total >= config.displacement_pairing_min_direction_ratio:
            deletions.append((index, _normalize_shape(deleted_pixels)))

    scores: dict[tuple[int, int], float] = {}
    for added_index, added_shape in additions:
        for deleted_index, deleted_shape in deletions:
            if not _geometry_matches(
                regions[added_index],
                regions[deleted_index],
                config.displacement_pairing_max_size_ratio,
            ):
                continue
            scores[(added_index, deleted_index)] = _translated_iou(
                added_shape, deleted_shape
            )

    best_deleted = _best_by_left(scores)
    best_added = _best_by_right(scores)
    pairs = {
        (added_index, deleted_index)
        for added_index, (deleted_index, score) in best_deleted.items()
        if score >= config.displacement_pairing_min_similarity
        and best_added.get(deleted_index, (-1, -1.0))[0] == added_index
    }
    if not pairs:
        return regions, 0

    paired_indices = {index for pair in pairs for index in pair}
    output = [region for index, region in enumerate(regions) if index not in paired_indices]
    for added_index, deleted_index in pairs:
        output.append(
            _union(
                regions[added_index],
                regions[deleted_index],
                scores[(added_index, deleted_index)],
            )
        )
    output.sort(key=lambda region: (region.y, region.x))
    return renumber_regions(output), len(pairs)


def _crop(mask: np.ndarray, region: DifferenceRegion, padding: int) -> np.ndarray:
    height, width = mask.shape[:2]
    x0 = max(0, region.x - padding)
    y0 = max(0, region.y - padding)
    x1 = min(width, region.x + region.width + padding)
    y1 = min(height, region.y + region.height + padding)
    return mask[y0:y1, x0:x1]


def _normalize_shape(mask: np.ndarray, size: int = 48) -> np.ndarray:
    points = cv2.findNonZero(mask.astype(np.uint8))
    canvas = np.zeros((size, size), dtype=bool)
    if points is None:
        return canvas
    x, y, width, height = cv2.boundingRect(points)
    cropped = mask[y : y + height, x : x + width].astype(np.uint8)
    scale = (size - 8) / max(width, height)
    resized = cv2.resize(
        cropped,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def _translated_iou(left: np.ndarray, right: np.ndarray, radius: int = 2) -> float:
    best = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(right, (dy, dx), axis=(0, 1))
            if dy > 0:
                shifted[:dy] = False
            elif dy < 0:
                shifted[dy:] = False
            if dx > 0:
                shifted[:, :dx] = False
            elif dx < 0:
                shifted[:, dx:] = False
            union = np.count_nonzero(left | shifted)
            if union:
                best = max(best, float(np.count_nonzero(left & shifted)) / union)
    return best


def _geometry_matches(
    left: DifferenceRegion, right: DifferenceRegion, max_ratio: float
) -> bool:
    width_ratio = max(left.width, right.width) / max(1, min(left.width, right.width))
    height_ratio = max(left.height, right.height) / max(1, min(left.height, right.height))
    return width_ratio <= max_ratio and height_ratio <= max_ratio


def _best_by_left(scores: dict[tuple[int, int], float]) -> dict[int, tuple[int, float]]:
    result: dict[int, tuple[int, float]] = {}
    for (left, right), score in scores.items():
        if score > result.get(left, (-1, -1.0))[1]:
            result[left] = (right, score)
    return result


def _best_by_right(scores: dict[tuple[int, int], float]) -> dict[int, tuple[int, float]]:
    result: dict[int, tuple[int, float]] = {}
    for (left, right), score in scores.items():
        if score > result.get(right, (-1, -1.0))[1]:
            result[right] = (left, score)
    return result


def _union(
    left: DifferenceRegion, right: DifferenceRegion, similarity: float
) -> DifferenceRegion:
    x0 = min(left.x, right.x)
    y0 = min(left.y, right.y)
    x1 = max(left.x + left.width, right.x + right.width)
    y1 = max(left.y + left.height, right.y + right.height)
    return replace(
        left,
        x=x0,
        y=y0,
        width=x1 - x0,
        height=y1 - y0,
        area=left.area + right.area,
        risk_reason="content_displacement",
        change_type="displaced",
        change_label="文本错位",
        classification_confidence=similarity,
        classification_reason="paired_displacement",
    )
