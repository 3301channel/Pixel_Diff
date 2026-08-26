"""Shared helpers for immutable difference regions."""

from __future__ import annotations

from dataclasses import replace

from pixel_diff.models import DifferenceRegion


def renumber_regions(regions: list[DifferenceRegion]) -> list[DifferenceRegion]:
    """Return regions numbered from one while preserving every metadata field."""
    return [replace(region, id=index) for index, region in enumerate(regions, start=1)]


def _rect_area(region: DifferenceRegion) -> int:
    """外接矩形面积（宽×高）。"""
    return region.width * region.height


def _rect_intersection_area(left: DifferenceRegion, right: DifferenceRegion) -> int:
    """两外接矩形的交集面积。"""
    x0 = max(left.x, right.x)
    y0 = max(left.y, right.y)
    x1 = min(left.x + left.width, right.x + right.width)
    y1 = min(left.y + left.height, right.y + right.height)
    return max(0, x1 - x0) * max(0, y1 - y0)


def deduplicate_overlapping_regions(
    regions: list[DifferenceRegion],
    overlap_ratio: float,
) -> list[DifferenceRegion]:
    """去除彼此重叠的差异区域，保留面积较大者、舍弃面积较小者。

    采用贪心 NMS：先把区域按外接矩形面积降序排列，依次把每个候选框与
    已保留的大框比对——若候选框被某个已保留框覆盖的比例（交集面积 / 候选框
    矩形面积）达到 ``overlap_ratio``，则丢弃候选框；否则保留。

    判定基于「小框被大框覆盖的比例」，而非双向 IoU，因此：
    - 一个小框完全落在大框内（覆盖比 1.0）→ 丢弃小框；
    - 两个并排大框仅边缘少量重叠（覆盖比很小）→ 都保留；
    - 面积相近的框即使部分重叠也不会被误删。

    去重后按 (y, x, 面积降序) 重新排序并重新编号，保持输出稳定。
    """
    if not regions or overlap_ratio <= 0:
        return list(regions)

    # 面积降序；面积相同时按坐标与 id 稳定排序，保证结果可复现
    ordered = sorted(regions, key=lambda r: (-_rect_area(r), r.y, r.x, r.id))

    kept: list[DifferenceRegion] = []
    for candidate in ordered:
        drop = False
        for anchor in kept:
            intersection = _rect_intersection_area(candidate, anchor)
            if intersection <= 0:
                continue
            candidate_area = _rect_area(candidate)
            if candidate_area > 0 and intersection / candidate_area >= overlap_ratio:
                drop = True
                break
        if not drop:
            kept.append(candidate)

    kept.sort(key=lambda r: (r.y, r.x, -_rect_area(r)))
    return renumber_regions(kept)

