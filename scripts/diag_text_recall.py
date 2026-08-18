"""Diagnose text-layer recall for the benchmark pair.

Prints row pairings and raw text-difference regions BEFORE merging/filtering.
"""
import fitz
from pixel_diff.models import PixelDiffConfig
from pixel_diff.text_layer import (
    _extract_pdf_rows, _paired_rows, extract_text_difference_regions,
    extract_sensitive_text_recall_regions
)
from pixel_diff.io import load_document_page_bgr
from pixel_diff.binarization import binarize_template_bgr, binarize_scan_bgr
import numpy as np, cv2

BASE = "artifacts/api_tasks/d9831e51329f4b46bd3908d381254451"
FA = f"{BASE}/inputs/file_a.pdf"
FB = f"{BASE}/inputs/file_b.pdf"
cfg = PixelDiffConfig.from_yaml("configs/sensitive_recall_trial.yaml")

for pg in range(4):
    print(f"\n================ PAGE {pg} ================")
    # raw rows
    trows = _extract_pdf_rows(FA, pg, cfg.dpi)
    srows = _extract_pdf_rows(FB, pg, cfg.dpi)
    print(f"template rows={len(trows)} scan rows={len(srows)}")
    print("--- TEMPLATE rows ---")
    for i, r in enumerate(trows):
        print(f"  [{i}] y={r.center_y:.0f} {r.text!r}")
    print("--- SCAN rows ---")
    for i, r in enumerate(srows):
        print(f"  [{i}] y={r.center_y:.0f} {r.text!r}")

    pairs = _paired_rows(FB, FA, pg, cfg.dpi)
    print("--- PAIRS ---")
    for sr, tr in pairs:
        print(f"  T(y={tr.center_y:.0f} {tr.text!r})  <->  S(y={sr.center_y:.0f} {sr.text!r})")

    h, w = load_document_page_bgr(FA, pg, cfg.dpi).shape[:2]
    txt_regs = extract_text_difference_regions(FB, FA, pg, (h, w), cfg)
    print(f"--- text-diff regions ({len(txt_regs)}) ---")
    for r in txt_regs:
        print(f"  @({r.x},{r.y})+{r.width}x{r.height} text={r.template_text!r}")

    # sensitive recall needs diff_mask
    tpl = load_document_page_bgr(FA, pg, cfg.dpi)
    scn = load_document_page_bgr(FB, pg, cfg.dpi)
    from pixel_diff.engine import PixelDiffEngine
    # quick align via engine helper not easily exposed; use identity-ish for density
    tb = binarize_template_bgr(tpl)
    sb = binarize_scan_bgr(scn, cfg)
    diff = cv2.bitwise_xor(tb, sb)
    sens_regs = extract_sensitive_text_recall_regions(FA, pg, diff, cfg)
    print(f"--- sensitive recall regions ({len(sens_regs)}) ---")
    for r in sens_regs:
        print(f"  @({r.x},{r.y})+{r.width}x{r.height} text={r.template_text!r}")
