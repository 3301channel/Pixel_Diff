"""Diagnostic: run full engine on the two referenced contract PDFs and dump every
region per page, plus raw XOR blue-residual area vs detected region area.

Usage:
    python scripts/diag_append_merge.py [start_page] [end_page] [config.yaml]
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import numpy as np
import cv2
import fitz

from pixel_diff.models import PixelDiffConfig
from pixel_diff.engine import PixelDiffEngine
from pixel_diff.io import load_document_page_bgr
from pixel_diff.binarization import binarize_template_bgr, binarize_scan_bgr
from pixel_diff.differ import xor_difference

TPL = r"C:/Users/Jason/Desktop/十二页合同测试文件.pdf"
SCAN = r"C:/Users/Jason/Desktop/十二页合同测试文件改动.pdf"

DEFAULT_CFG = "configs/sensitive_recall_trial.yaml"


def raw_xor_area(cfg, tpl_path, scan_path, page, dpi):
    """Render both pages, binarize, align roughly (identity), XOR -> blue residual count."""
    tpl = load_document_page_bgr(tpl_path, page, dpi)
    scn = load_document_page_bgr(scan_path, page, dpi)
    # simple identity align (digital vs digital, both same size assumed)
    h = min(tpl.shape[0], scn.shape[0]); w = min(tpl.shape[1], scn.shape[1])
    tpl = tpl[:h, :w]; scn = scn[:h, :w]
    tb = binarize_template_bgr(tpl)
    sb = binarize_scan_bgr(scn, cfg)
    diff = xor_difference(tb, sb)
    return int(cv2.countNonZero(diff))


def main() -> None:
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    cfg_path = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CFG

    cfg = PixelDiffConfig.from_yaml(cfg_path)
    engine = PixelDiffEngine(cfg)

    n = fitz.open(TPL).page_count
    end = min(end, n - 1)
    print(f"=== PDF pages={n}, config={cfg_path}, range={start}-{end} ===")

    out = {}
    for p in range(start, end + 1):
        t0 = time.time()
        try:
            res = engine.compare(SCAN, TPL, page=p)
            dt = time.time() - t0
            regs = res.differences
            # raw blue residual area (rough identity-aligned XOR)
            try:
                blue = raw_xor_area(cfg, TPL, SCAN, p, cfg.dpi)
            except Exception as e:  # noqa: BLE001
                blue = -1
                print(f"  [warn] blue area failed p{p}: {e}")
            detected_area = sum(int(r.width) * int(r.height) for r in regs)
            print(f"\n--- page {p} ({dt:.1f}s) regions={len(regs)} "
                  f"blue_xor={blue} detected_box_area={detected_area} ---")
            rows = []
            for r in regs:
                txt = (r.template_text or "").replace("\n", " ")
                print(f"  #{r.id:>2} {str(r.change_type):9} x={r.x:>4} y={r.y:>4} "
                      f"w={r.width:>4} h={r.height:>4} risk={str(r.risk_level):5} "
                      f"add={r.added_pixels} del={r.deleted_pixels} T={txt!r}")
                rows.append(dict(id=r.id, change_type=r.change_type, x=r.x, y=r.y,
                                 w=r.width, h=r.height, risk=r.risk_level,
                                 added=r.added_pixels, deleted=r.deleted_pixels,
                                 template_text=txt))
            ann = res.metadata.get("text_annotations", [])
            out[str(p)] = dict(regions=rows, blue_xor=blue,
                               detected_box_area=detected_area,
                               annotations=ann)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"\n!!! page {p} EXCEPTION: {e}")
            traceback.print_exc()
            out[str(p)] = dict(error=str(e))

    Path("scripts/diag_append_merge.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== wrote scripts/diag_append_merge.json ===")


if __name__ == "__main__":
    main()
