"""Run PixelDiffEngine with a single config variant and report misses/FP."""
import sys, dataclasses, fitz
from pixel_diff.engine import PixelDiffEngine
from pixel_diff.models import PixelDiffConfig

BASE = "artifacts/api_tasks/d9831e51329f4b46bd3908d381254451"
FA = f"{BASE}/inputs/file_a.pdf"
FB = f"{BASE}/inputs/file_b.pdf"
DPI = 300; ZOOM = DPI / 72.0

def line_boxes(path):
    doc = fitz.open(path)
    out = []
    for pi in range(doc.page_count):
        texts, boxes = [], []
        for block in doc[pi].get_text("dict")["blocks"]:
            if "lines" not in block: continue
            for line in block["lines"]:
                txt = "".join(sp["text"] for sp in line["spans"]).rstrip()
                if not txt: continue
                b = line["bbox"]
                texts.append(txt); boxes.append((int(b[0]*ZOOM), int(b[1]*ZOOM), int(b[2]*ZOOM), int(b[3]*ZOOM)))
        out.append((texts, boxes))
    doc.close()
    return out

def changed_lines(ta, tb):
    n = max(len(ta), len(tb))
    for i in range(n):
        a = ta[i] if i < len(ta) else ""
        b = tb[i] if i < len(tb) else ""
        if a != b: yield i

def intersect(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix1-ix0) * max(0, iy1-iy0) if ix1 > ix0 and iy1 > iy0 else 0

def main():
    cfg = PixelDiffConfig.from_yaml("configs/sensitive_recall_trial.yaml")
    variant = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if variant == "no_low_filter":
        cfg = dataclasses.replace(cfg, risk_review_filter_low=False)
    elif variant == "no_risk":
        cfg = dataclasses.replace(cfg, risk_review_enabled=False)
    elif variant == "min_change_1":
        cfg = dataclasses.replace(cfg, min_significant_pixel_change=1)
    elif variant == "no_low_and_min1":
        cfg = dataclasses.replace(cfg, risk_review_filter_low=False, min_significant_pixel_change=1)

    engine = PixelDiffEngine(cfg)
    per_page = []
    for p in range(4):
        res = engine.compare(FB, FA, page=p)
        per_page.append(res.differences)
        print(f"page {p}: {len(res.differences)} regions")

    A, B = line_boxes(FA), line_boxes(FB)
    total_miss = total_fp = 0
    for p in range(4):
        ta, ba = A[p]; tb, bb = B[p]
        changed = list(changed_lines(ta, tb))
        gt = [(max(0,bb[i][0]-8), max(0,bb[i][1]-8), bb[i][2]+8, bb[i][3]+8) for i in changed]
        eng = per_page[p]
        gt_cov = [False]*len(gt); reg_used = [False]*len(eng)
        for gi, g in enumerate(gt):
            for ri, r in enumerate(eng):
                rb = (r.x, r.y, r.x+r.width, r.y+r.height)
                if intersect(g, rb) > 0:
                    gt_cov[gi] = True; reg_used[ri] = True
        miss = sum(1 for x in gt_cov if not x)
        fp = sum(1 for x in reg_used if not x)
        total_miss += miss; total_fp += fp
        print(f"page {p}: changed={len(gt)} miss={miss} fp={fp}")
    print(f"TOTAL regions={sum(len(x) for x in per_page)} MISS={total_miss} FP={total_fp}")

if __name__ == "__main__":
    main()
