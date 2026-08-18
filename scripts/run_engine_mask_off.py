"""Run PixelDiffEngine on the benchmark pair with text-layer filter ON/OFF
and report the number of missed/false-positive regions (line-bbox based).
"""
import dataclasses
import fitz
from pixel_diff.engine import PixelDiffEngine
from pixel_diff.models import PixelDiffConfig

BASE = "artifacts/api_tasks/d9831e51329f4b46bd3908d381254451"
FA = f"{BASE}/inputs/file_a.pdf"
FB = f"{BASE}/inputs/file_b.pdf"
DPI = 300
ZOOM = DPI / 72.0

# line boxes from fitz dict (ground truth geometry)
def line_boxes(path):
    doc = fitz.open(path)
    out = []
    for pi in range(doc.page_count):
        texts, boxes = [], []
        for block in doc[pi].get_text("dict")["blocks"]:
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

def changed_lines(ta, tb):
    n = max(len(ta), len(tb))
    for i in range(n):
        a = ta[i] if i < len(ta) else ""
        b = tb[i] if i < len(tb) else ""
        if a != b:
            yield i

def intersect(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return (ix1 - ix0) * (iy1 - iy0)

def evaluate(per_page_regions):
    A = line_boxes(FA); B = line_boxes(FB)
    total_miss = 0; total_fp = 0
    for p in range(len(A)):
        ta, ba = A[p]; tb, bb = B[p]
        changed = list(changed_lines(ta, tb))
        gt = [(max(0,bb[i][0]-8), max(0,bb[i][1]-8), bb[i][2]+8, bb[i][3]+8) for i in changed]
        eng = per_page_regions[p]
        gt_cov = [False]*len(gt); reg_used = [False]*len(eng)
        for gi, g in enumerate(gt):
            for ri, r in enumerate(eng):
                rb = (r.x, r.y, r.x+r.width, r.y+r.height)
                if intersect(g, rb) > 0:
                    gt_cov[gi] = True; reg_used[ri] = True
        total_miss += sum(1 for x in gt_cov if not x)
        total_fp += sum(1 for x in reg_used if not x)
    return total_miss, total_fp

def run(label, cfg):
    print(f"\n=== {label} ===")
    engine = PixelDiffEngine(cfg)
    per_page = []
    for p in range(4):
        res = engine.compare(FB, FA, page=p)
        per_page.append(res.differences)
        print(f"  page {p} regions={len(res.differences)}")
    miss, fp = evaluate(per_page)
    total = sum(len(x) for x in per_page)
    print(f"  TOTAL regions={total} MISS={miss} FP={fp}")
    return per_page, miss, fp

base_cfg = PixelDiffConfig.from_yaml("configs/sensitive_recall_trial.yaml")
configs = [
    ("baseline", base_cfg),
    ("risk_review_filter_low=false", dataclasses.replace(base_cfg, risk_review_filter_low=False)),
    ("risk_review_enabled=false", dataclasses.replace(base_cfg, risk_review_enabled=False)),
    ("min_significant_pixel_change=1", dataclasses.replace(base_cfg, min_significant_pixel_change=1)),
    ("mask-off", dataclasses.replace(base_cfg, pdf_text_layer_filter=False)),
]
for name, c in configs:
    try:
        run(name, c)
    except Exception as e:
        print(f"ERROR in {name}: {e}")
        import traceback; traceback.print_exc()
