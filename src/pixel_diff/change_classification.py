"""Classify final difference regions from directional foreground residuals."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pixel_diff.models import DifferenceRegion, PixelDiffConfig

_LABELS = {
    "added": "新增",
    "deleted": "删除",
    "modified": "修改",
    "displaced": "文本错位",
}


def classify_difference_regions(
    regions: list[DifferenceRegion],
    scan_binary: np.ndarray,
    template_binary: np.ndarray,
    config: PixelDiffConfig,
) -> list[DifferenceRegion]:
    if not config.difference_classification_enabled:
        return regions
    height, width = scan_binary.shape[:2]
    scan_foreground = scan_binary == 0
    template_foreground = template_binary == 0
    added_mask = scan_foreground & ~template_foreground
    deleted_mask = template_foreground & ~scan_foreground
    output: list[DifferenceRegion] = []
    for region in regions:
        x0, y0 = max(0, region.x), max(0, region.y)
        x1 = min(width, region.x + region.width)
        y1 = min(height, region.y + region.height)
        added = int(np.count_nonzero(added_mask[y0:y1, x0:x1]))
        deleted = int(np.count_nonzero(deleted_mask[y0:y1, x0:x1]))
        total = added + deleted
        if region.change_type == "displaced":
            output.append(replace(region, added_pixels=added, deleted_pixels=deleted))
            continue
        if total == 0:
            change_type, confidence = "modified", 0.0
            reason = "empty_direction_modified"
        else:
            added_ratio = added / total
            deleted_ratio = deleted / total
            if added_ratio >= config.difference_direction_ratio_threshold:
                change_type, confidence = "added", added_ratio
                reason = "directional_added"
            elif deleted_ratio >= config.difference_direction_ratio_threshold:
                change_type, confidence = "deleted", deleted_ratio
                reason = "directional_deleted"
            else:
                change_type = "modified"
                confidence = 1.0 - abs(added - deleted) / total
                reason = "bidirectional_modified"
                upgrade = _large_displacement_confidence(
                    region, added, deleted, width, height, config
                )
                if upgrade is not None:
                    change_type = "displaced"
                    confidence = upgrade
                    reason = "large_balanced_horizontal_residual"
        output.append(
            replace(
                region,
                change_type=change_type,
                change_label=_LABELS[change_type],
                added_pixels=added,
                deleted_pixels=deleted,
                classification_confidence=float(confidence),
                classification_reason=reason,
            )
        )
    return output


def _large_displacement_confidence(
    region: DifferenceRegion,
    added: int,
    deleted: int,
    page_width: int,
    page_height: int,
    config: PixelDiffConfig,
) -> float | None:
    if not config.large_modified_as_displaced_enabled or added == 0 or deleted == 0:
        return None
    area_ratio = (region.width * region.height) / max(1, page_width * page_height)
    aspect_ratio = region.width / max(1, region.height)
    direction_balance = min(added, deleted) / max(added, deleted)
    if (
        area_ratio < config.large_modified_min_page_area_ratio
        or aspect_ratio < config.large_modified_min_aspect_ratio
        or direction_balance < config.large_modified_min_direction_balance
    ):
        return None
    return min(
        1.0,
        area_ratio / config.large_modified_min_page_area_ratio,
        aspect_ratio / config.large_modified_min_aspect_ratio,
        direction_balance / config.large_modified_min_direction_balance,
    )
