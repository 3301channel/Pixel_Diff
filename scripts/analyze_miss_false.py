"""Cross-check engine regions (diff_result.json) against TRUE pixel-diff blobs.

漏判 (missed): a real diff blob with no overlapping engine region.
误判 (false positive): an engine region overlapping no real diff blob.
"""
import json, sys
import numpy as np
import cv2
import fitz

BASE = "artifacts/api_tasks/d9831e51329f4b46bd3908d381254451"
RES = f"{BASE}/outputs/d9831e51329f4b46bd3908d381254451_file_a_vs_file_b/diff_result.json"
FA = f"{BASE}/inputs/file_a.pdf"
FB = f"{BASE}/inputs/file_b.pdf"
DPI = 300
ZOOM = DPI / 72.0

def render(path, page):
    doc = fitz.open(path)
    pg = doc[page]
    pix = pg.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img

def sift_align(tpl, scan):
    """Align scan to tpl via SIFT homography (mirrors engine)."""
    tg = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    sg = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(tg, None)
    k2, d2 = sift.detectAndCompute(sg, None)
    if d1 is None or d2 is None or len(d1) < 4 or len(d2) < 4:
        return scan, 0.0
    bf = cv2.BFMatcher()
    raw = bf.knnMatch(d1, d2, k=2)
    good = []
    for m, n in raw:
        if m.distance < 0.75 * n.distance:
            good.append((m, n))
    if len(good) < 10:
        return scan, 0.0
    src = np.float32([k2[m.trainIdx].pt for m, _ in good])
    dst = np.float32([k1[m.queryIdx].pt for m, _ in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return scan, 0.0
    h, w = tpl.shape[:2]
    aligned = cv2.warpPerspective(scan, H, (w, h),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    inl = mask.sum() / max(1, len(mask))
    return aligned, float(inl)

def binarize(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu per-image
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return b

def true_blobs(tpl, scan):
    aligned, inl = sift_align(tpl, scan)
    tb = binarize(tpl)
    sb = binarize(aligned)
    # crop edges (engine crops 40)
    m = 40
    tb[:m, :] = 0; tb[-m:, :] = 0; tb[:, :m] = 0; tb[:, -m:] = 0
    sb[:m, :] = 0; sb[-m:, :] = 0; sb[:, :m] = 0; sb[:, -m:] = 0
    diff = cv2.bitwise_xor(tb, sb)
    # light clean
    k = np.ones((3, 3), np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 25:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h, int(a)))
    return boxes, inl, diff

def overlap_area(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx); iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
    if ix2 <= ix or iy2 <= iy:
        return 0
    return (ix2 - ix) * (iy2 - iy)

def main():
    d = json.load(open(RES, encoding="utf-8"))
    regions = d["regions"]
    # engine page is 1-based PDF page+1
    dt = fitz.open(FA); dpages = dt.page_count; dt.close()

    miss = []   # true blobs with no covering region
    falsepos = []  # regions covering no blob
    for p in range(dpages):
        tpl = render(FA, p)
        scan = render(FB, p)
        blobs, inl, _ = true_blobs(tpl, scan)
        eng = [r for r in regions if r["page"] == p + 1]
        # match
        blob_covered = [False] * len(blobs)
        reg_used = [False] * len(eng)
        for bi, b in enumerate(blobs):
            for ri, r in enumerate(eng):
                rb = (r["x"], r["y"], r["width"], r["height"])
                ov = overlap_area(b[:4], rb)
                if ov > 0.3 * b[4] or ov > 0.3 * (r["width"] * r["height"]):
                    blob_covered[bi] = True
                    reg_used[ri] = True
        for bi, b in enumerate(blobs):
            if not blob_covered[bi]:
                miss.append((p + 1, b))
        for ri, r in enumerate(eng):
            if not reg_used[ri]:
                falsepos.append((p + 1, r))
        print(f"page {p+1}: inlier={inl:.3f} true_blobs={len(blobs)} eng_regions={len(eng)} "
              f"miss={sum(1 for x in blob_covered if not x)} falsepos={sum(1 for x in reg_used if not x)}")

    print("\n===== 漏判 (real diff blobs not covered by any region) =====")
    for pg, b in miss:
        print(f"  p{pg} blob @({b[0]},{b[1]})+{b[2]}x{b[3]} area={b[4]}")
    print("\n===== 误判 (engine region covering no real diff blob) =====")
    for pg, r in falsepos:
        print(f"  p{pg} #{r['id']} {r['change_type']}/{r['risk_level']} "
              f"@({r['x']},{r['y']})+{r['width']}x{r['height']} A={r['template_text']!r}")

if __name__ == "__main__":
    main()
