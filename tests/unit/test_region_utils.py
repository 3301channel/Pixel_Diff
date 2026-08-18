from pixel_diff.models import DifferenceRegion


def test_renumber_regions_preserves_optional_metadata() -> None:
    from pixel_diff.region_utils import renumber_regions

    region = DifferenceRegion(
        id=99,
        x=1,
        y=2,
        width=3,
        height=4,
        area=5.0,
        risk_level="HIGH",
        risk_reason="sensitive_template_text_overlap",
        template_text="010-6528",
        ocr_text="010-6529",
        sensitive_type="amount_or_number_symbol",
        kept=True,
    )

    result = renumber_regions([region])

    assert result == [
        DifferenceRegion(
            id=1,
            x=1,
            y=2,
            width=3,
            height=4,
            area=5.0,
            risk_level="HIGH",
            risk_reason="sensitive_template_text_overlap",
            template_text="010-6528",
            ocr_text="010-6529",
            sensitive_type="amount_or_number_symbol",
            kept=True,
        )
    ]

