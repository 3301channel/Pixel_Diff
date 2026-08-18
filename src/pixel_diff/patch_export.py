"""Export candidate patches for neural-network assisted review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pixel_diff.models import DifferenceRegion, PixelDiffConfig


def export_region_patches(
    regions: list[DifferenceRegion],
    template_bgr: np.ndarray,
    scan_bgr: np.ndarray,
    output_dir: str | Path,
    page: int,
    config: PixelDiffConfig,
) -> Path | None:
    """Write template/scan/diff patches and metadata for model training."""

    if not config.patch_export_enabled or not regions:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "page": page + 1,
        "padding": config.patch_export_padding,
        "regions": [],
    }
    for region in regions:
        crop = _crop_bounds(template_bgr.shape[:2], region, config.patch_export_padding)
        x0, y0, x1, y1 = crop
        template_patch = template_bgr[y0:y1, x0:x1]
        scan_patch = scan_bgr[y0:y1, x0:x1]
        diff_patch = cv2.absdiff(template_patch, scan_patch)

        prefix = f"page_{page + 1:04d}_region_{region.id:04d}"
        template_name = f"{prefix}_template.png"
        scan_name = f"{prefix}_scan.png"
        diff_name = f"{prefix}_diff.png"
        _write_png(out / template_name, template_patch)
        _write_png(out / scan_name, scan_patch)
        _write_png(out / diff_name, diff_patch)

        region_payload: dict[str, Any] = dict(region.to_dict())
        region_payload.update(
            {
                "crop": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "template_patch": template_name,
                "scan_patch": scan_name,
                "diff_patch": diff_name,
                "label": "",
            }
        )
        metadata["regions"].append(region_payload)

    with (out / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return out


def _crop_bounds(
    image_shape: tuple[int, int],
    region: DifferenceRegion,
    padding: int,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    x0 = max(0, region.x - padding)
    y0 = max(0, region.y - padding)
    x1 = min(width, region.x + region.width + padding)
    y1 = min(height, region.y + region.height + padding)
    return x0, y0, x1, y1


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write patch image: {path}")
