"""Experiment: prove unchanged_text_mask halo (padding) swallows appended chars.

For each known missed change, render+align the page, XOR-diff, then apply
build_unchanged_text_mask at several paddings and count surviving diff pixels
inside the appended-char bounding box.
"""
import json
import dataclasses
import numpy as np
import cv2
import fitz
from pixel_diff.models import PixelDiffConfig
from pixel_diff.text_layer import build_unchanged_text_mask
from pixel_diff.binarization import binarize_template_bgr, binarize_scan_bgr

BASE = "artifacts/api_tasks/d9831e51329f4b46bd3908d381254451"
FA = f"{BASE}/inputs/file_a.pdf"
FB = f"{BASE}/inputs/file_b.pdf"
DPI = 300
ZOOM = DPI / 72.0

# (page, appended-char bbox in 300dpi px) from the missed-change analysis
MISSED = [
    (0, (1082, 1404, 1187, 1465)),   # 个 -> 1 个  ('1 ' prepended)
    (1, (367, 691, 1103, 753)),      # 仓库...as  ('as' appended at right end)
    (2, (1235, 2250, 1335, 2311)),   # 个 -> 个... ('...' appended)
]

def render(path, page):
    doc = fitz.open(path)
    pix = doc[page].get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if pix.n == 3 else cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    doc.close()
    return img

def sift_align(tpl, scan):
    tg = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    sg = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(tg, None)
    k2, d2 = sift.detectAndCompute(sg, None)
    if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
        return scan
    bf = cv2.BFMatcher()
    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return scan
    src = np.float32([k2[m.trainIdx].pt for m in good])
    dst = np.float32([k1[m.queryIdx].pt for m in good])
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return scan
    h, w = tpl.shape[:2]
    return cv2.warpPerspective(scan, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def diff_mask(tpl, aligned):
    tb = binarize_template_bgr(tpl)
    sb = binarize_scan_bgr(aligned, cfg)
    m = 40
    tb[:m, :] = 0; tb[-m:, :] = 0; tb[:, :m] = 0; tb[:, -m:] = 0
    sb[:m, :] = 0; sb[-m:, :] = 0; sb[:, :m] = 0; sb[:, -m:] = 0
    d = cv2.bitwise_xor(tb, sb)
    k = np.ones((3, 3), np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, k)
    return d

def count_in(box, mask):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(mask.shape[1], x1), min(mask.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return 0
    return int(cv2.countNonZero(mask[y0:y1, x0:x1]))

cfg = PixelDiffConfig.from_yaml("configs/sensitive_recall_trial.yaml")

for pg, box in MISSED:
    tpl = render(FA, pg)
    scan = render(FB, pg)
    aligned = sift_align(tpl, scan)
    raw = diff_mask(tpl, aligned)
    base = count_in(box, raw)
    print(f"\npage {pg+1} appended-char box={box}  RAW diff pixels={base}")
    for pad in (5, 2, 1, 0):
        c = dataclasses.replace(cfg, pdf_text_mask_padding=pad)
        mask = build_unchanged_text_mask(FB, FA, pg, raw.shape, c)
        if mask is None:
            print(f"  padding={pad}: mask=None (skipped)")
            continue
        masked = cv2.bitwise_and(raw, mask)
        print(f"  padding={pad}: surviving diff pixels in box = {count_in(box, masked)}")
