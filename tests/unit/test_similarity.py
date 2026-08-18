import numpy as np

from pixel_diff.models import DifferenceRegion


def test_ssim_returns_one_for_identical_images() -> None:
    from pixel_diff.similarity import ssim

    image = np.arange(400, dtype=np.uint8).reshape(20, 20)

    assert ssim(image, image) == 1.0


def test_best_ssim_finds_bounded_translation() -> None:
    from pixel_diff.similarity import best_ssim_for_region, shift_gray

    template = np.full((40, 40), 255, dtype=np.uint8)
    template[12:25, 14:22] = 0
    scan = shift_gray(template, 2, -1)
    region = DifferenceRegion(id=1, x=8, y=8, width=22, height=22, area=100.0)

    score = best_ssim_for_region(
        region,
        scan_gray=scan,
        template_gray=template,
        padding=2,
        search_radius=3,
    )

    assert score > 0.99


def test_best_ssim_returns_zero_for_tiny_crop() -> None:
    from pixel_diff.similarity import best_ssim_for_region

    image = np.zeros((10, 10), dtype=np.uint8)
    region = DifferenceRegion(id=1, x=0, y=0, width=2, height=2, area=4.0)

    assert best_ssim_for_region(region, image, image, padding=0, search_radius=1) == 0.0


def test_cached_template_ssim_search_matches_legacy_score() -> None:
    from pixel_diff.similarity import (
        best_ssim_for_region,
        best_ssim_for_region_cached,
    )

    generator = np.random.default_rng(314159)
    template = generator.integers(0, 256, size=(75, 90), dtype=np.uint8)
    scan = np.roll(template, shift=(2, -3), axis=(0, 1))
    region = DifferenceRegion(id=1, x=6, y=5, width=70, height=58, area=2000.0)

    legacy = best_ssim_for_region(region, scan, template, padding=4, search_radius=4)
    cached = best_ssim_for_region_cached(region, scan, template, padding=4, search_radius=4)

    assert cached == legacy


def test_best_ssim_can_stop_after_zero_shift_reaches_threshold(monkeypatch) -> None:
    import pixel_diff.similarity as similarity

    image = np.arange(1600, dtype=np.uint8).reshape(40, 40)
    region = DifferenceRegion(id=1, x=5, y=5, width=25, height=25, area=625.0)
    calls = 0
    original_ssim = similarity.ssim

    def counted_ssim(first: np.ndarray, second: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return original_ssim(first, second)

    monkeypatch.setattr(similarity, "ssim", counted_ssim)
    score = similarity.best_ssim_for_region(
        region,
        image,
        image,
        padding=2,
        search_radius=4,
        stop_at=0.95,
    )

    assert score == 1.0
    assert calls == 1
