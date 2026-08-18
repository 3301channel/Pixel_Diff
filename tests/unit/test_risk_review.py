from __future__ import annotations

import cv2
import numpy as np
import pytest

from pixel_diff.models import DifferenceRegion, PixelDiffConfig
from pixel_diff.risk_review import _ocr_region_text, _text_from_rapidocr_result, apply_risk_review


def test_risk_review_converts_page_images_to_gray_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aligned = np.full((120, 180, 3), 255, dtype=np.uint8)
    template = aligned.copy()
    regions = [
        DifferenceRegion(id=1, x=10, y=10, width=30, height=30, area=900.0),
        DifferenceRegion(id=2, x=60, y=60, width=30, height=30, area=900.0),
    ]
    config = PixelDiffConfig(
        risk_review_enabled=True,
        risk_review_plain_text_residual_enabled=True,
    )
    real_cvt_color = cv2.cvtColor
    calls = 0

    def counting_cvt_color(image: np.ndarray, code: int) -> np.ndarray:
        nonlocal calls
        calls += 1
        return real_cvt_color(image, code)

    monkeypatch.setattr(cv2, "cvtColor", counting_cvt_color)

    apply_risk_review(
        regions,
        aligned_bgr=aligned,
        template_text_by_region={1: "普通", 2: "文本"},
        config=config,
        template_bgr=template,
    )

    assert calls == 2


def test_risk_review_marks_sensitive_template_text_as_high() -> None:
    regions = [DifferenceRegion(id=1, x=10, y=10, width=40, height=20, area=800.0)]
    config = PixelDiffConfig(risk_review_enabled=True)

    result = apply_risk_review(
        regions=regions,
        aligned_bgr=np.full((80, 160, 3), 255, dtype=np.uint8),
        template_text_by_region={1: "ICBKCN8J"},
        config=config,
    )

    assert len(result.regions) == 1
    assert result.regions[0].risk_level == "HIGH"
    assert result.regions[0].risk_reason == "sensitive_template_text_overlap"
    assert result.regions[0].template_text == "ICBKCN8J"
    assert result.regions[0].sensitive_type == "latin_digit"
    assert result.metrics["risk_review_high"] == 1


def test_risk_review_can_filter_low_risk_regions_without_template_text() -> None:
    regions = [
        DifferenceRegion(id=1, x=10, y=10, width=40, height=20, area=800.0),
        DifferenceRegion(id=2, x=80, y=30, width=30, height=20, area=600.0),
    ]
    config = PixelDiffConfig(risk_review_enabled=True, risk_review_filter_low=True)

    result = apply_risk_review(
        regions=regions,
        aligned_bgr=np.full((80, 160, 3), 255, dtype=np.uint8),
        template_text_by_region={2: "普通文本"},
        config=config,
    )

    assert [region.id for region in result.regions] == [1]
    assert result.regions[0].risk_level == "MEDIUM"
    assert result.regions[0].template_text == "普通文本"
    assert result.metrics["risk_review_low_filtered"] == 1


def test_risk_review_keeps_large_visual_residual_without_template_text() -> None:
    region = DifferenceRegion(id=1, x=20, y=20, width=140, height=62, area=8680.0)
    config = PixelDiffConfig(risk_review_enabled=True, risk_review_filter_low=True)

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=np.full((140, 220, 3), 255, dtype=np.uint8),
        template_text_by_region={},
        config=config,
    )

    assert len(result.regions) == 1
    assert result.regions[0].risk_level == "MEDIUM"
    assert result.regions[0].risk_reason == "large_visual_residual_without_text"
    assert result.metrics["risk_review_medium"] == 1
    assert result.metrics["risk_review_low_filtered"] == 0


def test_risk_review_filters_large_low_contrast_watermark_without_template_text() -> None:
    template = np.full((140, 220, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.putText(scan, "WATER", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (190, 190, 190), 4)
    region = DifferenceRegion(id=1, x=10, y=30, width=180, height=70, area=12600.0)
    config = PixelDiffConfig(risk_review_enabled=True, risk_review_filter_low=True)

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=scan,
        template_bgr=template,
        template_text_by_region={},
        config=config,
    )

    assert result.regions == []
    assert result.metrics["risk_review_low"] == 1
    assert result.metrics["risk_review_low_filtered"] == 1


def test_risk_review_downgrades_plain_text_when_structure_is_similar() -> None:
    template = np.full((120, 260, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.putText(template, "PLAIN", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(scan, "PLAIN", (31, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    region = DifferenceRegion(id=1, x=26, y=42, width=120, height=40, area=1800.0)
    config = PixelDiffConfig(
        risk_review_enabled=True,
        risk_review_plain_text_residual_enabled=True,
        risk_review_plain_text_ssim_threshold=0.70,
    )

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=scan,
        template_bgr=template,
        template_text_by_region={1: "普通正文"},
        config=config,
    )

    assert result.regions[0].risk_level == "LOW"
    assert result.regions[0].risk_reason == "plain_text_structural_residual"
    assert result.metrics["risk_review_low"] == 1


def test_risk_review_keeps_sensitive_text_high_even_when_structure_is_similar() -> None:
    image = np.full((80, 160, 3), 255, dtype=np.uint8)
    region = DifferenceRegion(id=1, x=10, y=10, width=40, height=20, area=800.0)
    config = PixelDiffConfig(
        risk_review_enabled=True,
        risk_review_plain_text_residual_enabled=True,
        risk_review_plain_text_ssim_threshold=0.70,
    )

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=image,
        template_bgr=image,
        template_text_by_region={1: "A123"},
        config=config,
    )

    assert result.regions[0].risk_level == "HIGH"
    assert result.regions[0].risk_reason == "sensitive_template_text_overlap"


def test_risk_review_does_not_downgrade_protected_cjk_confusable_text() -> None:
    template = np.full((120, 260, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.putText(template, "TEXT", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(scan, "TEXT", (31, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    region = DifferenceRegion(id=1, x=26, y=42, width=120, height=40, area=1800.0)
    config = PixelDiffConfig(
        risk_review_enabled=True,
        risk_review_plain_text_residual_enabled=True,
        risk_review_plain_text_ssim_threshold=0.70,
    )

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=scan,
        template_bgr=template,
        template_text_by_region={1: "使甲"},
        config=config,
    )

    assert result.regions[0].risk_level == "MEDIUM"
    assert result.regions[0].risk_reason == "template_text_overlap"


def test_risk_review_filters_high_confidence_same_shape_stroke_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = np.full((140, 360, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.putText(template, "SAME", (45, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)
    cv2.putText(scan, "SAME", (47, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    region = DifferenceRegion(id=1, x=40, y=42, width=150, height=55, area=4200.0)
    config = PixelDiffConfig(
        risk_review_enabled=True,
        risk_review_filter_low=True,
        risk_review_stroke_match_enabled=True,
        risk_review_stroke_match_tolerance=4,
        risk_review_stroke_match_min_coverage=0.90,
        risk_review_stroke_match_max_area=6500,
    )
    monkeypatch.setattr("pixel_diff.risk_review._stroke_match_ocr_text", lambda *_: "由")

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=scan,
        template_bgr=template,
        template_text_by_region={1: "由"},
        config=config,
    )

    assert result.regions == []
    assert result.metrics["risk_review_stroke_match_filtered"] == 1


def test_risk_review_keeps_changed_shape_despite_stroke_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = np.full((140, 360, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.putText(template, "SAME", (45, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)
    cv2.putText(scan, "SAVE", (45, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2)
    region = DifferenceRegion(id=1, x=40, y=42, width=150, height=55, area=4200.0)
    config = PixelDiffConfig(
        risk_review_enabled=True,
        risk_review_filter_low=True,
        risk_review_stroke_match_enabled=True,
        risk_review_stroke_match_tolerance=3,
        risk_review_stroke_match_min_coverage=0.96,
        risk_review_stroke_match_max_area=6500,
    )
    monkeypatch.setattr("pixel_diff.risk_review._stroke_match_ocr_text", lambda *_: "甲")

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=scan,
        template_bgr=template,
        template_text_by_region={1: "由"},
        config=config,
    )

    assert len(result.regions) == 1


def _narrow_stroke_images(*, shifted_scan: bool = False) -> tuple[np.ndarray, np.ndarray]:
    template = np.full((100, 120, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.line(template, (55, 40), (55, 62), (0, 0, 0), 2)
    scan_x = 68 if shifted_scan else 55
    cv2.line(scan, (scan_x, 40), (scan_x, 62), (0, 0, 0), 2)
    return scan, template


def _narrow_stroke_config(**overrides: object) -> PixelDiffConfig:
    values: dict[str, object] = {
        "risk_review_enabled": True,
        "risk_review_filter_low": True,
        "risk_review_plain_text_residual_enabled": False,
        "risk_review_narrow_stroke_enabled": True,
        "risk_review_narrow_stroke_max_area": 500.0,
        "risk_review_narrow_stroke_max_width": 20,
        "risk_review_narrow_stroke_max_height": 32,
        "risk_review_narrow_stroke_min_coverage": 0.96,
    }
    values.update(overrides)
    return PixelDiffConfig(**values)


def test_risk_review_filters_narrow_high_coverage_stroke_residual() -> None:
    scan, template = _narrow_stroke_images()
    region = DifferenceRegion(id=1, x=48, y=40, width=17, height=23, area=352.0)

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=scan,
        template_bgr=template,
        template_text_by_region={1: "\u6c49"},
        config=_narrow_stroke_config(),
    )

    assert result.regions == []
    assert result.metrics["risk_review_narrow_stroke_filtered"] == 1


@pytest.mark.parametrize(
    ("region", "template_text", "shifted_scan", "config_overrides"),
    [
        (
            DifferenceRegion(id=1, x=48, y=40, width=17, height=23, area=352.0),
            "\u6c49",
            True,
            {},
        ),
        (
            DifferenceRegion(id=1, x=48, y=40, width=21, height=23, area=352.0),
            "\u6c49",
            False,
            {},
        ),
        (
            DifferenceRegion(id=1, x=48, y=40, width=17, height=33, area=352.0),
            "\u6c49",
            False,
            {},
        ),
        (
            DifferenceRegion(id=1, x=48, y=40, width=17, height=23, area=501.0),
            "\u6c49",
            False,
            {},
        ),
        (
            DifferenceRegion(id=1, x=48, y=40, width=17, height=23, area=352.0),
            "123",
            False,
            {},
        ),
        (
            DifferenceRegion(id=1, x=48, y=40, width=17, height=23, area=352.0),
            "\u6c49",
            False,
            {"risk_review_narrow_stroke_enabled": False},
        ),
    ],
    ids=["low-coverage", "too-wide", "too-tall", "too-large", "sensitive", "disabled"],
)
def test_risk_review_keeps_narrow_stroke_when_any_safety_gate_fails(
    region: DifferenceRegion,
    template_text: str,
    shifted_scan: bool,
    config_overrides: dict[str, object],
) -> None:
    scan, template = _narrow_stroke_images(shifted_scan=shifted_scan)

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=scan,
        template_bgr=template,
        template_text_by_region={1: template_text},
        config=_narrow_stroke_config(**config_overrides),
    )

    assert len(result.regions) == 1
    assert result.metrics["risk_review_narrow_stroke_filtered"] == 0


def test_risk_review_filters_unchanged_bottom_centered_page_number() -> None:
    template = np.full((500, 400, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.putText(template, "4", (190, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(scan, "4", (191, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    region = DifferenceRegion(id=1, x=185, y=445, width=35, height=40, area=900.0)
    config = PixelDiffConfig(
        risk_review_enabled=True,
        risk_review_filter_low=True,
        risk_review_page_number_match_enabled=True,
        risk_review_stroke_match_tolerance=4,
        risk_review_stroke_match_min_coverage=0.90,
    )

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=scan,
        template_bgr=template,
        template_text_by_region={1: "4"},
        config=config,
    )

    assert result.regions == []
    assert result.metrics["risk_review_page_number_filtered"] == 1


def test_risk_review_keeps_changed_bottom_page_number() -> None:
    template = np.full((500, 400, 3), 255, dtype=np.uint8)
    scan = template.copy()
    cv2.putText(template, "4", (190, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(scan, "7", (190, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    region = DifferenceRegion(id=1, x=185, y=445, width=35, height=40, area=900.0)
    config = PixelDiffConfig(
        risk_review_enabled=True,
        risk_review_filter_low=True,
        risk_review_page_number_match_enabled=True,
        risk_review_stroke_match_tolerance=2,
        risk_review_stroke_match_min_coverage=0.95,
    )

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=scan,
        template_bgr=template,
        template_text_by_region={1: "4"},
        config=config,
    )

    assert len(result.regions) == 1


def test_risk_review_filters_plain_text_when_ocr_matches_template(
    monkeypatch,
) -> None:
    image = np.full((120, 220, 3), 255, dtype=np.uint8)
    region = DifferenceRegion(id=1, x=60, y=50, width=20, height=14, area=280.0)
    config = PixelDiffConfig(
        risk_review_enabled=True,
        risk_review_filter_low=True,
        risk_review_ocr_enabled=True,
    )

    monkeypatch.setattr("pixel_diff.risk_review._ocr_region_text", lambda *_: "程参")

    result = apply_risk_review(
        regions=[region],
        aligned_bgr=image,
        template_text_by_region={1: "程参"},
        config=config,
    )

    assert result.regions == []
    assert result.metrics["risk_review_low"] == 1
    assert result.metrics["risk_review_low_filtered"] == 1


def test_ocr_region_text_uses_dedicated_ocr_padding(monkeypatch) -> None:
    image = np.full((120, 120, 3), 255, dtype=np.uint8)
    region = DifferenceRegion(id=1, x=50, y=50, width=10, height=10, area=100.0)
    config = PixelDiffConfig(
        risk_review_ocr_enabled=True,
        risk_review_region_padding=2,
        risk_review_ocr_padding=12,
    )
    patch_shapes: list[tuple[int, int, int]] = []

    def fake_rapidocr(patch: np.ndarray) -> str:
        patch_shapes.append(patch.shape)
        return "程参"

    monkeypatch.setattr("pixel_diff.risk_review._read_with_pytesseract", lambda _: "")
    monkeypatch.setattr("pixel_diff.risk_review._read_with_rapidocr", fake_rapidocr)
    monkeypatch.setattr("pixel_diff.risk_review._read_with_easyocr", lambda _: "")

    assert _ocr_region_text(region, image, config) == "程参"
    assert patch_shapes == [(34, 34, 3)]


def test_text_from_rapidocr_result_accepts_legacy_tuple_output() -> None:
    result = (
        [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "888", 0.92),
            ([[20, 0], [30, 0], [30, 10], [20, 10]], "999", 0.90),
        ],
        0.12,
    )

    assert _text_from_rapidocr_result(result) == "888999"


def test_text_from_rapidocr_result_accepts_object_with_txts() -> None:
    class Result:
        txts = ("张", "三")

    assert _text_from_rapidocr_result(Result()) == "张三"
