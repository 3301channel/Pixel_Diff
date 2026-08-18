"""Authoritative miss/false-positive check using fitz line geometry as ground truth.

For each page we compute the text diff between file_a and file_b (done externally),
then locate each CHANGED line's bounding box in file_b via fitz. An engine region
covers a change if its box intersects the changed-line box (within tolerance).
  - 漏判 (missed): a changed line with no covering engine region.
  - 误判 (false positive): an engine region intersecting no changed line.
"""
import json
import fitz
import numpy as np

BASE = "artifacts/api_tasks/d9831e51329f4b46bd3908d381254451"
RES = f"{BASE}/outputs/d9831e51329f4b46bd3908d381254451_file_a_vs_file_b/diff_result.json"
FA = f"{BASE}/inputs/file_a.pdf"
FB = f"{BASE}/inputs/file_b.pdf"
DPI = 300
ZOOM = DPI / 72.0

def line_boxes(path):
    """Return list of (pageno, [line texts], [line bboxes]) using fitz dict."""
    doc = fitz.open(path)
    out = []
    for pi in range(doc.page_count):
        pg = doc[pi]
        d = pg.get_text("dict")
        texts, boxes = [], []
        for block in d["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                txt = "".join(sp["text"] for sp in line["spans"]).rstrip()
                if not txt:
                    continue
                b = line["bbox"]
                texts.append(txt)
                boxes.append((int(b[0]*ZOOM), int(b[1]*ZOOM), int(b[2]*ZOOM), int(b[3]*ZOOM)))
        out.append((texts, boxes))
    doc.close()
    return out

def changed_lines(ta_lines, tb_lines):
    """Yield indices of lines that differ between A and B (per page)."""
    n = max(len(ta_lines), len(tb_lines))
    for i in range(n):
        a = ta_lines[i] if i < len(ta_lines) else ""
        b = tb_lines[i] if i < len(tb_lines) else ""
        if a != b:
            yield i

def intersect(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return (ix1 - ix0) * (iy1 - iy0)

def main():
    d = json.load(open(RES, encoding="utf-8"))
    regions = d["regions"]
    A = line_boxes(FA)
    B = line_boxes(FB)
    TOL = 8  # px tolerance for bbox edge match

    all_miss = []
    all_fp = []
    for p in range(len(A)):
        ta, ba = A[p]
        tb, bb = B[p]
        changed = list(changed_lines(ta, tb))
        # changed-line boxes in B (padded slightly)
        gt_boxes = []
        for i in changed:
            x0, y0, x1, y1 = bb[i]
            a_txt = ta[i] if i < len(ta) else "<none>"
            b_txt = tb[i] if i < len(tb) else "<none>"
            gt_boxes.append((x0 - TOL, y0 - TOL, x1 + TOL, y1 + TOL, a_txt, b_txt))
        eng = [r for r in regions if r["page"] == p + 1]
        # mark coverage
        gt_cov = [False]*len(gt_boxes)
        reg_used = [False]*len(eng)
        for gi, g in enumerate(gt_boxes):
            for ri, r in enumerate(eng):
                rb = (r["x"], r["y"], r["x"]+r["width"], r["y"]+r["height"])
                if intersect(g[:4], rb) > 0:
                    gt_cov[gi] = True
                    reg_used[ri] = True
        for gi, g in enumerate(gt_boxes):
            if not gt_cov[gi]:
                all_miss.append((p+1, g))
        for ri, r in enumerate(eng):
            if not reg_used[ri]:
                all_fp.append((p+1, r))
        print(f"page {p+1}: changed_lines={len(changed)} gt_boxes={len(gt_boxes)} "
              f"eng={len(eng)} MISS={sum(1 for x in gt_cov if not x)} FP={sum(1 for x in reg_used if not x)}")

    print("\n===== 漏判 (changed lines with NO covering region) =====")
    for pg, g in all_miss:
        print(f"  p{pg} A={g[4]!r}\n       B={g[5]!r}  box=({g[0]},{g[1]})-({g[2]},{g[3]})")
    print("\n===== 误判 (region intersecting NO changed line) =====")
    for pg, r in all_fp:
        print(f"  p{pg} #{r['id']} {r['change_type']}/{r['risk_level']} "
              f"@({r['x']},{r['y']})+{r['width']}x{r['height']} A={r['template_text']!r}")

if __name__ == "__main__":
    main()
