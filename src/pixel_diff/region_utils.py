"""Shared helpers for immutable difference regions."""

from __future__ import annotations

from dataclasses import replace

from pixel_diff.models import DifferenceRegion


def renumber_regions(regions: list[DifferenceRegion]) -> list[DifferenceRegion]:
    """Return regions numbered from one while preserving every metadata field."""
    return [replace(region, id=index) for index, region in enumerate(regions, start=1)]

