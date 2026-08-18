"""Authoritative verification: drive fixed engine, compare against
SequenceMatcher-aligned fitz line-geometry ground truth."""
import dataclasses, difflib, fitz
from pixel_diff.engine import PixelDiffEngine
from pixel_diff.models import PixelDiffConfig

BASE = "artifacts/api_tasks/d9831e51329f4b46bd3908d381254451"
FA = f"{BASE}/inputs/file_a.pdf"
FB = f"{BASE}/inputs/file_b.pdf"
DPI = 300
ZOOM = DPI / 72.0

def line_boxes(path):
    doc = fitz.open(path)
    out = []
    for pi in range(doc.page_count):
        texts, boxes = [], []
        for block in doc[pi].get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                txt = "".join(sp["text"] for sp in line["spans"])
                b = line["bbox"]
                texts.append(txt)
                boxes.append((int(b[0]*ZOOM), int(b[1]*ZOOM), int(b[2]*ZOOM), int(b[3]*ZOOM)))
        out.append((texts, boxes))
    doc.close()
    return out

def intersect(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0, ix1-ix0) * max(0, iy1-iy0) if ix1 > ix0 and iy1 > iy0 else 0

def main():
    cfg = PixelDiffConfig.from_yaml("configs/sensitive_recall_trial.yaml")
    engine = PixelDiffEngine(cfg)
    per_page = []
    for p in range(4):
        res = engine.compare(FB, FA, page=p)
        per_page.append(res.differences)
        print(f"page {p}: {len(res.differences)} regions")

    A, B = line_boxes(FA), line_boxes(FB)
    TOL = 8
    total_miss = total_fp = 0
    for p in range(4):
        ta, ba = A[p]; tb, bb = B[p]
        sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
        gt_boxes = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            for idx in range(j1, j2):
                x0, y0, x1, y1 = bb[idx]
                gt_boxes.append((max(0,x0-TOL), max(0,y0-TOL), x1+TOL, y1+TOL,
                                 ta[idx] if idx < len(ta) else "<none>", tb[idx]))
        eng = per_page[p]
        gt_cov = [False]*len(gt_boxes); reg_used = [False]*len(eng)
        for gi, g in enumerate(gt_boxes):
            for ri, r in enumerate(eng):
                rb = (r.x, r.y, r.x+r.width, r.y+r.height)
                if intersect(g[:4], rb) > 0:
                    gt_cov[gi] = True; reg_used[ri] = True
        miss = sum(1 for x in gt_cov if not x)
        fp = sum(1 for x in reg_used if not x)
        total_miss += miss; total_fp += fp
        print(f"page {p}: changed={len(gt_boxes)} miss={miss} fp={fp}")
        for gi, g in enumerate(gt_boxes):
            if not gt_cov[gi]:
                print(f"   >>> MISS p{p+1} A={g[4]!r} B={g[5]!r} box=({g[0]},{g[1]})-({g[2]},{g[3]})")
        for ri, r in enumerate(eng):
            if not reg_used[ri]:
                print(f"   >>> FP   p{p+1} {r.change_type}/{r.risk_level} @({r.x},{r.y})+{r.width}x{r.height} tpl={r.template_text!r}")
    print(f"\nTOTAL regions={sum(len(x) for x in per_page)} MISS={total_miss} FP={total_fp}")

if __name__ == "__main__":
    main()
