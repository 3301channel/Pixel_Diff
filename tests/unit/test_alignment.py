from __future__ import annotations

import numpy as np
import pytest

from pixel_diff import AlignmentError, PixelDiffConfig
from pixel_diff.alignment import align_scan_to_template_bgr


def test_alignment_rejects_blank_pages_without_descriptors() -> None:
    blank = np.full((120, 120, 3), 255, dtype=np.uint8)

    with pytest.raises(AlignmentError):
        align_scan_to_template_bgr(blank, blank, PixelDiffConfig(min_good_matches=4))
