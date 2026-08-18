"""Constrained dense local displacement compensation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from pixel_diff.models import PixelDiffConfig


@dataclass(frozen=True)
class LocalWarpResult:
    aligned_bgr: np.ndarray
    applied: bool
    max_displacement: float
    mean_displacement: float
    gate_skipped: bool = False
    gate_foreground_iou: float = 0.0


def apply_constrained_local_warp_bgr(
    scan_bgr: np.ndarray,
    template_bgr: np.ndarray,
    config: PixelDiffConfig,
) -> LocalWarpResult:
    """Apply optional dense optical-flow warp with hard displacement clipping."""
    if not config.local_warp_enabled or config.local_warp_max_displacement == 0:
        return LocalWarpResult(scan_bgr, False, 0.0, 0.0)

    height, width = template_bgr.shape[:2]
    if scan_bgr.shape[:2] != (height, width):
        return LocalWarpResult(scan_bgr, False, 0.0, 0.0)

    scan_gray = cv2.cvtColor(scan_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    gate_iou = (
        _foreground_iou(scan_gray, template_gray)
        if config.local_warp_gate_enabled
        else 0.0
    )
    if config.local_warp_gate_enabled and gate_iou >= config.local_warp_gate_min_iou:
        return LocalWarpResult(
            scan_bgr,
            False,
            0.0,
            0.0,
            gate_skipped=True,
            gate_foreground_iou=gate_iou,
        )

    scale = config.local_warp_scale
    if scale < 1.0:
        small_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        scan_flow_gray = cv2.resize(scan_gray, small_size, interpolation=cv2.INTER_AREA)
        template_flow_gray = cv2.resize(template_gray, small_size, interpolation=cv2.INTER_AREA)
    else:
        scan_flow_gray = scan_gray
        template_flow_gray = template_gray

    initial_flow = np.zeros((*scan_flow_gray.shape[:2], 2), dtype=np.float32)
    flow = cv2.calcOpticalFlowFarneback(
        scan_flow_gray,
        template_flow_gray,
        initial_flow,
        pyr_scale=0.5,
        levels=3,
        winsize=31,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow_array: NDArray[np.float32] = np.asarray(flow, dtype=np.float32)
    if scale < 1.0:
        flow_array = np.asarray(
            cv2.resize(flow_array, (width, height), interpolation=cv2.INTER_LINEAR),
            dtype=np.float32,
        )
        flow_array = np.asarray(flow_array / np.float32(scale), dtype=np.float32)

    flow_array = _smooth_flow(flow_array, config.local_warp_blur_kernel)
    flow_array = _clip_flow(flow_array, float(config.local_warp_max_displacement))
    magnitude = np.sqrt(
        flow_array[..., 0] * flow_array[..., 0] + flow_array[..., 1] * flow_array[..., 1]
    )
    max_displacement = float(np.max(magnitude))
    mean_displacement = float(np.mean(magnitude))
    if max_displacement == 0.0:
        return LocalWarpResult(
            scan_bgr, False, 0.0, 0.0, gate_foreground_iou=gate_iou
        )

    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    aligned = cv2.remap(
        scan_bgr,
        grid_x - flow_array[..., 0].astype(np.float32),
        grid_y - flow_array[..., 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return LocalWarpResult(
        aligned,
        True,
        max_displacement,
        mean_displacement,
        gate_foreground_iou=gate_iou,
    )


def _foreground_iou(scan_gray: np.ndarray, template_gray: np.ndarray) -> float:
    """Estimate page agreement cheaply on quarter-resolution foreground masks."""
    height, width = template_gray.shape[:2]
    size = (max(1, width // 4), max(1, height // 4))
    scan_small = cv2.resize(scan_gray, size, interpolation=cv2.INTER_AREA)
    template_small = cv2.resize(template_gray, size, interpolation=cv2.INTER_AREA)
    _, scan_binary = cv2.threshold(
        scan_small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    _, template_binary = cv2.threshold(
        template_small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    scan_foreground = scan_binary != 0
    template_foreground = template_binary != 0
    union = np.count_nonzero(scan_foreground | template_foreground)
    if union == 0:
        return 1.0
    intersection = np.count_nonzero(scan_foreground & template_foreground)
    return float(intersection / union)


def _smooth_flow(flow: np.ndarray, blur_kernel: int) -> np.ndarray:
    if blur_kernel == 0:
        return flow
    smoothed = np.empty_like(flow)
    smoothed[..., 0] = cv2.GaussianBlur(flow[..., 0], (blur_kernel, blur_kernel), 0)
    smoothed[..., 1] = cv2.GaussianBlur(flow[..., 1], (blur_kernel, blur_kernel), 0)
    return smoothed


def _clip_flow(flow: np.ndarray, max_displacement: float) -> np.ndarray:
    if max_displacement == 0:
        return np.zeros_like(flow)
    magnitude = np.sqrt(flow[..., 0] * flow[..., 0] + flow[..., 1] * flow[..., 1])
    scale = np.ones_like(magnitude, dtype=np.float32)
    over_limit = magnitude > max_displacement
    scale[over_limit] = max_displacement / magnitude[over_limit]
    clipped = np.empty_like(flow)
    clipped[..., 0] = flow[..., 0] * scale
    clipped[..., 1] = flow[..., 1] * scale
    return clipped
