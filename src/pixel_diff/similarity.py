"""Shared grayscale similarity helpers."""

from __future__ import annotations

import cv2
import numpy as np

from pixel_diff.models import DifferenceRegion


def best_ssim_for_region(
    region: DifferenceRegion,
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    padding: int,
    search_radius: int,
    stop_at: float | None = None,
) -> float:
    """Return the best SSIM around a bounded translation search."""
    height, width = template_gray.shape[:2]
    x0 = max(0, region.x - padding)
    y0 = max(0, region.y - padding)
    x1 = min(width, region.x + region.width + padding)
    y1 = min(height, region.y + region.height + padding)
    scan_crop = scan_gray[y0:y1, x0:x1]
    template_crop = template_gray[y0:y1, x0:x1]
    if scan_crop.size < 100 or template_crop.size < 100:
        return 0.0

    best_score = ssim(template_crop, scan_crop)
    if stop_at is not None and best_score >= stop_at:
        return best_score
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0:
                continue
            best_score = max(best_score, ssim(template_crop, shift_gray(scan_crop, dx, dy)))
            if stop_at is not None and best_score >= stop_at:
                return best_score
    return best_score


def best_ssim_for_region_cached(
    region: DifferenceRegion,
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    padding: int,
    search_radius: int,
    stop_at: float | None = None,
) -> float:
    """Return the same bounded SSIM search while reusing template moments."""
    height, width = template_gray.shape[:2]
    x0 = max(0, region.x - padding)
    y0 = max(0, region.y - padding)
    x1 = min(width, region.x + region.width + padding)
    y1 = min(height, region.y + region.height + padding)
    scan_crop = scan_gray[y0:y1, x0:x1]
    template_crop = template_gray[y0:y1, x0:x1]
    if scan_crop.size < 100 or template_crop.size < 100:
        return 0.0

    template_stats = _template_ssim_stats(template_crop)
    best_score = _ssim_with_template_stats(template_stats, scan_crop)
    if stop_at is not None and best_score >= stop_at:
        return best_score
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = shift_gray(scan_crop, dx, dy)
            best_score = max(
                best_score,
                _ssim_with_template_stats(template_stats, shifted),
            )
            if stop_at is not None and best_score >= stop_at:
                return best_score
    return best_score


def shift_gray(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a grayscale image using the existing interpolation semantics."""
    height, width = image.shape[:2]
    transform = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Compute structural similarity using the existing constants."""
    return _ssim_with_template_stats(_template_ssim_stats(first), second)


def _template_ssim_stats(
    first: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_f = first.astype(np.float64)
    mu1 = cv2.GaussianBlur(first_f, (7, 7), 1.5)
    mu1_sq = mu1 * mu1
    sigma1_sq = cv2.GaussianBlur(first_f * first_f, (7, 7), 1.5) - mu1_sq
    return first_f, mu1, mu1_sq, sigma1_sq


def _ssim_with_template_stats(
    stats: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    second: np.ndarray,
) -> float:
    first_f, mu1, mu1_sq, sigma1_sq = stats
    second_f = second.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu2 = cv2.GaussianBlur(second_f, (7, 7), 1.5)
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma2_sq = cv2.GaussianBlur(second_f * second_f, (7, 7), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(first_f * second_f, (7, 7), 1.5) - mu1_mu2
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float(np.clip((numerator / denominator).mean(), 0.0, 1.0))
