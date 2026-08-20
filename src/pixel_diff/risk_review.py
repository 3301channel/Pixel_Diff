"""Risk review for candidate difference boxes."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Literal, cast

import cv2
import numpy as np

from pixel_diff.models import DifferenceRegion, PixelDiffConfig
from pixel_diff.similarity import best_ssim_for_region

RiskLevel = Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]


@dataclass(frozen=True)
class RiskReviewResult:
    regions: list[DifferenceRegion]
    metrics: dict[str, int]


def apply_risk_review(
    regions: list[DifferenceRegion],
    aligned_bgr: np.ndarray,
    template_text_by_region: dict[int, str],
    config: PixelDiffConfig,
    template_bgr: np.ndarray | None = None,
    scan_gray: np.ndarray | None = None,
    template_gray: np.ndarray | None = None,
) -> RiskReviewResult:
    """Annotate regions with risk levels and optionally filter LOW-risk boxes."""

    if not config.risk_review_enabled or not regions:
        return RiskReviewResult(regions=regions, metrics=_empty_metrics())

    if scan_gray is None:
        scan_gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    if template_gray is None and template_bgr is not None:
        template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    reviewed_regions: list[DifferenceRegion] = []
    metrics = _empty_metrics()
    for region in regions:
        template_text = template_text_by_region.get(region.id, "") or region.template_text or ""
        ocr_text = _ocr_region_text(region, aligned_bgr, config)
        if _ocr_confirms_template_text(template_text, ocr_text, config):
            level: RiskLevel = "LOW"
            reason = "ocr_matches_template_text"
            sensitive_type = None
        else:
            level, reason, sensitive_type = _classify_risk(template_text, ocr_text)
        if (
            level == "LOW"
            and not _is_background_watermark_residual(
                region,
                scan_gray=scan_gray,
                template_gray=template_gray,
                config=config,
            )
            and _is_large_visual_residual(region, config)
        ):
            level = "MEDIUM"
            reason = "large_visual_residual_without_text"
        # 文本层差异区域：SequenceMatcher 已确认模板/扫描文字不同，
        # 因此不应当被注册残差、结构残差等启发式规则降级为 LOW。
        if not getattr(region, "text_layer_protected", False):
            if _is_plain_text_structural_residual(
                region,
                template_text=template_text,
                ocr_text=ocr_text,
                sensitive_type=sensitive_type,
                scan_gray=scan_gray,
                template_gray=template_gray,
                config=config,
            ):
                level = "LOW"
                reason = "plain_text_structural_residual"
            if _is_same_shape_text_residual(
                region,
                template_text=template_text,
                ocr_text=ocr_text,
                aligned_bgr=aligned_bgr,
                sensitive_type=sensitive_type,
                scan_gray=scan_gray,
                template_gray=template_gray,
                config=config,
            ):
                level = "LOW"
                reason = "same_shape_stroke_residual"
                metrics["risk_review_stroke_match_filtered"] += 1
            if _is_narrow_stroke_registration_residual(
                region,
                template_text=template_text,
                sensitive_type=sensitive_type,
                scan_gray=scan_gray,
                template_gray=template_gray,
                config=config,
            ):
                level = "LOW"
                reason = "narrow_stroke_registration_residual"
                metrics["risk_review_narrow_stroke_filtered"] += 1
            if _is_unchanged_page_number(
                region,
                template_text=template_text,
                ocr_text=ocr_text,
                scan_gray=scan_gray,
                template_gray=template_gray,
                config=config,
            ):
                level = "LOW"
                reason = "unchanged_page_number"
                metrics["risk_review_page_number_filtered"] += 1
            # 未重叠像素差异：只保留字符形，横线/竖线/噪点降级为 LOW
            if level == "MEDIUM" and reason == "visual_difference_without_text":
                if _is_line_or_speckle_residual(region, config):
                    level = "LOW"
                    reason = "line_or_speckle_residual"
        # 文本层差异区域最低保留 MEDIUM，避免被 risk_review_filter_low 误杀。
        if getattr(region, "text_layer_protected", False) and level == "LOW":
            level = "MEDIUM"
            reason = "text_layer_difference"
        metrics[f"risk_review_{level.lower()}"] += 1
        kept = not (config.risk_review_filter_low and level == "LOW")
        if not kept:
            metrics["risk_review_low_filtered"] += 1
            continue
        reviewed_regions.append(
            DifferenceRegion(
                id=len(reviewed_regions) + 1,
                x=region.x,
                y=region.y,
                width=region.width,
                height=region.height,
                area=region.area,
                risk_level=level,
                risk_reason=reason,
                template_text=template_text or None,
                ocr_text=ocr_text or None,
                sensitive_type=sensitive_type,
                kept=True,
                text_layer_protected=region.text_layer_protected,
            )
        )

    return RiskReviewResult(regions=reviewed_regions, metrics=metrics)


def _is_large_visual_residual(region: DifferenceRegion, config: PixelDiffConfig) -> bool:
    if not config.risk_review_large_visual_residual_enabled:
        return False
    return (
        region.area >= config.risk_review_large_visual_min_area
        and region.width >= config.risk_review_large_visual_min_width
        and region.height >= config.risk_review_large_visual_min_height
    )


def _is_background_watermark_residual(
    region: DifferenceRegion,
    *,
    scan_gray: np.ndarray,
    template_gray: np.ndarray | None,
    config: PixelDiffConfig,
) -> bool:
    if not config.risk_review_watermark_filter_enabled:
        return False
    if template_gray is None:
        return False
    if not _is_large_visual_residual(region, config):
        return False

    scan_crop, template_crop = _crop_gray_pair(scan_gray, template_gray, region, padding=8)
    if scan_crop.size == 0 or template_crop.size == 0:
        return False

    abs_delta = cv2.absdiff(scan_crop, template_crop)
    p95_delta = float(np.percentile(abs_delta, 95))
    very_dark_ratio = float((scan_crop < config.risk_review_watermark_dark_threshold).mean())
    template_dark_ratio = float(
        (template_crop < config.risk_review_watermark_template_dark_threshold).mean()
    )
    return (
        p95_delta <= config.risk_review_watermark_max_p95_delta
        and very_dark_ratio <= config.risk_review_watermark_max_very_dark_ratio
        and template_dark_ratio <= config.risk_review_watermark_max_template_dark_ratio
    )


def _crop_gray_pair(
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    region: DifferenceRegion,
    padding: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = template_gray.shape[:2]
    x0 = max(0, region.x - padding)
    y0 = max(0, region.y - padding)
    x1 = min(width, region.x + region.width + padding)
    y1 = min(height, region.y + region.height + padding)
    return scan_gray[y0:y1, x0:x1], template_gray[y0:y1, x0:x1]


def _is_plain_text_structural_residual(
    region: DifferenceRegion,
    *,
    template_text: str,
    ocr_text: str,
    sensitive_type: str | None,
    scan_gray: np.ndarray,
    template_gray: np.ndarray | None,
    config: PixelDiffConfig,
) -> bool:
    if not config.risk_review_plain_text_residual_enabled:
        return False
    if template_gray is None:
        return False
    if not _normalize_text(template_text) or _normalize_text(ocr_text):
        return False
    if sensitive_type is not None:
        return False
    if _has_protected_plain_text_char(template_text, config):
        return False
    if region.area > config.risk_review_plain_text_max_area:
        return False

    score = best_ssim_for_region(
        region,
        scan_gray=scan_gray,
        template_gray=template_gray,
        padding=config.risk_review_ssim_padding,
        search_radius=config.risk_review_ssim_search_radius,
        stop_at=(
            config.risk_review_plain_text_ssim_threshold
            if config.ssim_early_exit_enabled
            else None
        ),
    )
    return score >= config.risk_review_plain_text_ssim_threshold


def _is_same_shape_text_residual(
    region: DifferenceRegion,
    *,
    template_text: str,
    ocr_text: str,
    aligned_bgr: np.ndarray,
    sensitive_type: str | None,
    scan_gray: np.ndarray,
    template_gray: np.ndarray | None,
    config: PixelDiffConfig,
) -> bool:
    if not config.risk_review_stroke_match_enabled or template_gray is None:
        return False
    if not _normalize_text(template_text) or _normalize_text(ocr_text):
        return False
    if sensitive_type is not None or not (
        config.risk_review_stroke_match_min_area
        <= region.area
        <= config.risk_review_stroke_match_max_area
    ):
        return False
    coverage = _stroke_match_coverage(
        region,
        scan_gray,
        template_gray,
        config,
    )
    if coverage < config.risk_review_stroke_match_min_coverage:
        return False
    candidate_text = _stroke_match_ocr_text(region, aligned_bgr, config)
    return _texts_confirm_same(
        template_text,
        candidate_text,
        config.risk_review_ocr_match_ratio,
    )


def _is_unchanged_page_number(
    region: DifferenceRegion,
    *,
    template_text: str,
    ocr_text: str,
    scan_gray: np.ndarray,
    template_gray: np.ndarray | None,
    config: PixelDiffConfig,
) -> bool:
    if not config.risk_review_page_number_match_enabled or template_gray is None:
        return False
    normalized = _normalize_text(template_text)
    normalized_ocr = _normalize_text(ocr_text)
    if not normalized.isdigit() or (normalized_ocr and normalized_ocr != normalized):
        return False
    height, width = template_gray.shape
    center_x = region.x + region.width / 2.0
    if region.y < height * config.risk_review_page_number_bottom_ratio:
        return False
    if abs(center_x - width / 2.0) > width * config.risk_review_page_number_center_tolerance_ratio:
        return False
    if (
        region.width > config.risk_review_page_number_max_width
        or region.height > config.risk_review_page_number_max_height
    ):
        return False
    return _normalized_stroke_match_coverage(
        region,
        scan_gray,
        template_gray,
        config,
    ) >= config.risk_review_page_number_min_coverage


def _is_narrow_stroke_registration_residual(
    region: DifferenceRegion,
    *,
    template_text: str,
    sensitive_type: str | None,
    scan_gray: np.ndarray,
    template_gray: np.ndarray | None,
    config: PixelDiffConfig,
) -> bool:
    if not config.risk_review_narrow_stroke_enabled or template_gray is None:
        return False
    if not _normalize_text(template_text) or sensitive_type is not None:
        return False
    if (
        region.area > config.risk_review_narrow_stroke_max_area
        or region.width > config.risk_review_narrow_stroke_max_width
        or region.height > config.risk_review_narrow_stroke_max_height
    ):
        return False
    return (
        _stroke_match_coverage(region, scan_gray, template_gray, config)
        >= config.risk_review_narrow_stroke_min_coverage
    )


def _is_line_or_speckle_residual(
    region: DifferenceRegion,
    config: PixelDiffConfig,
) -> bool:
    """判定「未重叠」像素差异是否为横线/竖线/噪点（而非字符）。

    对 visual_difference_without_text 区域做形状过滤：
    - 宽高比过大（width/height > max_aspect_ratio）→ 横线（表格线等）
    - 宽高比过小（width/height < min_aspect_ratio）→ 竖线（表格线等）
    - 包围盒面积过小（width*height < min_area）→ 噪点
    命中即视为噪声，返回 True（调用方降级为 LOW 过滤掉）。
    """
    if not config.risk_review_char_shape_filter_enabled:
        return False
    if region.width <= 0 or region.height <= 0:
        return True
    if region.width * region.height < config.risk_review_char_min_area:
        return True
    aspect = region.width / region.height
    if aspect > config.risk_review_char_max_aspect_ratio:
        return True
    if aspect < config.risk_review_char_min_aspect_ratio:
        return True
    return False


def _stroke_match_coverage(
    region: DifferenceRegion,
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    config: PixelDiffConfig,
) -> float:
    scan_crop, template_crop = _crop_gray_pair(
        scan_gray,
        template_gray,
        region,
        config.risk_review_stroke_match_padding,
    )
    if scan_crop.size < 100 or template_crop.size < 100:
        return 0.0
    scan_fg = _otsu_foreground(scan_crop)
    template_fg = _otsu_foreground(template_crop)
    if not np.any(scan_fg) or not np.any(template_fg):
        return 0.0
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
    tolerance = config.risk_review_stroke_match_tolerance
    template_coverage = float(np.mean(distance_to_scan[template_fg] <= tolerance))
    scan_coverage = float(np.mean(distance_to_template[scan_fg] <= tolerance))
    return min(template_coverage, scan_coverage)


def _otsu_foreground(gray: np.ndarray) -> np.ndarray:
    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return binary > 0


def _normalized_stroke_match_coverage(
    region: DifferenceRegion,
    scan_gray: np.ndarray,
    template_gray: np.ndarray,
    config: PixelDiffConfig,
) -> float:
    scan_crop, template_crop = _crop_gray_pair(
        scan_gray,
        template_gray,
        region,
        config.risk_review_stroke_match_padding,
    )
    scan_fg = _normalize_foreground_shape(_otsu_foreground(scan_crop))
    template_fg = _normalize_foreground_shape(_otsu_foreground(template_crop))
    if scan_fg is None or template_fg is None:
        return 0.0
    distance_to_scan = cv2.distanceTransform(
        np.where(scan_fg, 0, 255).astype(np.uint8), cv2.DIST_L2, 3
    )
    distance_to_template = cv2.distanceTransform(
        np.where(template_fg, 0, 255).astype(np.uint8), cv2.DIST_L2, 3
    )
    tolerance = config.risk_review_page_number_shape_tolerance
    template_coverage = float(np.mean(distance_to_scan[template_fg] <= tolerance))
    scan_coverage = float(np.mean(distance_to_template[scan_fg] <= tolerance))
    return min(template_coverage, scan_coverage)


def _normalize_foreground_shape(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    cropped = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return cv2.resize(
        cropped.astype(np.uint8),
        (64, 64),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _stroke_match_ocr_text(
    region: DifferenceRegion,
    aligned_bgr: np.ndarray,
    config: PixelDiffConfig,
) -> str:
    patch = _crop_region(aligned_bgr, region, config.risk_review_ocr_padding)
    if patch.size == 0:
        return ""
    text = _read_with_pytesseract(patch)
    if text:
        return text
    text = _read_with_rapidocr(patch)
    if text:
        return text
    return _read_with_easyocr(patch)


def _texts_confirm_same(template_text: str, candidate_text: str, ratio: float) -> bool:
    template = _normalize_text(template_text)
    candidate = _normalize_text(candidate_text)
    if not template or not candidate:
        return False
    if template in candidate or candidate in template:
        return True
    return SequenceMatcher(None, template, candidate).ratio() >= ratio


def _classify_risk(
    template_text: str,
    ocr_text: str,
) -> tuple[RiskLevel, str, str | None]:
    normalized_template = _normalize_text(template_text)
    normalized_ocr = _normalize_text(ocr_text)
    if normalized_template and normalized_ocr and normalized_template != normalized_ocr:
        sensitive_type = _sensitive_type(normalized_template) or _sensitive_type(normalized_ocr)
        return "HIGH", "template_text_differs_from_ocr", sensitive_type

    if normalized_template:
        sensitive_type = _sensitive_type(normalized_template)
        if sensitive_type is not None:
            return "HIGH", "sensitive_template_text_overlap", sensitive_type
        return "MEDIUM", "template_text_overlap", None

    if normalized_ocr:
        sensitive_type = _sensitive_type(normalized_ocr)
        if sensitive_type is not None:
            return "HIGH", "ocr_detected_sensitive_text_without_template_overlap", sensitive_type
        return "MEDIUM", "ocr_detected_text_without_template_overlap", None

    # 配准后既无模板文字重叠也无OCR重叠的像素差异，认定为差异（而非噪声），不再降级为LOW被过滤
    return "MEDIUM", "visual_difference_without_text", None


def _sensitive_type(text: str) -> str | None:
    has_digit = bool(re.search(r"\d", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    has_symbol = bool(re.search(r"[\s.,/\\\-()（）\[\]【】{}:：;；%￥¥$#*+&@_]", text))
    if has_digit and has_latin:
        return "latin_digit"
    if has_digit and has_symbol:
        return "amount_or_number_symbol"
    if has_digit:
        return "number"
    if has_latin:
        return "latin"
    if has_symbol:
        return "symbol"
    return None


def _has_protected_plain_text_char(text: str, config: PixelDiffConfig) -> bool:
    protected_chars = set(config.risk_review_plain_text_protected_chars)
    return any(char in protected_chars for char in text)


def _normalize_text(text: str) -> str:
    return text.strip()


def _ocr_region_text(
    region: DifferenceRegion,
    aligned_bgr: np.ndarray,
    config: PixelDiffConfig,
) -> str:
    if not config.risk_review_ocr_enabled:
        return ""

    patch = _crop_region(aligned_bgr, region, config.risk_review_ocr_padding)
    if patch.size == 0:
        return ""

    text = _read_with_pytesseract(patch)
    if text:
        return text
    text = _read_with_rapidocr(patch)
    if text:
        return text
    return _read_with_easyocr(patch)


def _crop_region(image: np.ndarray, region: DifferenceRegion, padding: int) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(0, region.x - padding)
    y0 = max(0, region.y - padding)
    x1 = min(width, region.x + region.width + padding)
    y1 = min(height, region.y + region.height + padding)
    return image[y0:y1, x0:x1]


def _ocr_confirms_template_text(
    template_text: str,
    ocr_text: str,
    config: PixelDiffConfig,
) -> bool:
    if not config.risk_review_ocr_match_filter_enabled:
        return False
    normalized_template = _normalize_text(template_text)
    normalized_ocr = _normalize_text(ocr_text)
    if not normalized_template or not normalized_ocr:
        return False
    if (
        _sensitive_type(normalized_template) is not None
        or _sensitive_type(normalized_ocr) is not None
    ):
        return False
    if _has_protected_plain_text_char(normalized_template, config):
        return False
    if normalized_template in normalized_ocr or normalized_ocr in normalized_template:
        return True
    ratio = SequenceMatcher(None, normalized_template, normalized_ocr).ratio()
    return ratio >= config.risk_review_ocr_match_ratio


def _read_with_pytesseract(patch: np.ndarray) -> str:
    try:
        import pytesseract  # type: ignore[import-not-found]
    except Exception:
        return ""
    try:
        return str(pytesseract.image_to_string(patch, config="--psm 7")).strip()
    except Exception:
        return ""


def _read_with_rapidocr(patch: np.ndarray) -> str:
    try:
        engine = _rapidocr_engine()
        if engine is None:
            return ""
        result = engine(patch)
        return _text_from_rapidocr_result(result)
    except Exception:
        return ""


@lru_cache(maxsize=1)
def _rapidocr_engine() -> Callable[[np.ndarray], object] | None:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return None
    try:
        return cast(Callable[[np.ndarray], object], RapidOCR())
    except Exception:
        return None


def _text_from_rapidocr_result(result: object) -> str:
    txts = getattr(result, "txts", None)
    if txts is not None:
        return "".join(str(text) for text in txts).strip()

    entries: object = result
    if isinstance(result, tuple) and result:
        entries = result[0]
    if not isinstance(entries, list):
        return ""

    texts: list[str] = []
    for entry in entries:
        entry_value = cast(Any, entry)
        if isinstance(entry_value, (list, tuple)) and len(entry_value) >= 2:
            texts.append(str(entry_value[1]))
    return "".join(texts).strip()


def _read_with_easyocr(patch: np.ndarray) -> str:
    try:
        import easyocr  # type: ignore[import-not-found]
    except Exception:
        return ""
    try:
        reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        return "".join(str(item) for item in reader.readtext(patch, detail=0)).strip()
    except Exception:
        return ""


def _empty_metrics() -> dict[str, int]:
    return {
        "risk_review_high": 0,
        "risk_review_medium": 0,
        "risk_review_low": 0,
        "risk_review_unknown": 0,
        "risk_review_low_filtered": 0,
        "risk_review_stroke_match_filtered": 0,
        "risk_review_narrow_stroke_filtered": 0,
        "risk_review_page_number_filtered": 0,
    }
