"""Fixed miss/FP check: keep empty lines and align A/B lines with SequenceMatcher."""
import json
import fitz
import difflib

BASE = "artifacts/api_tasks/d9831e51329f4b46bd3908d381254451"
RES = f"{BASE}/outputs/d9831e51329f4b46bd3908d381254451_file_a_vs_file_b/diff_result.json"
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
    d = json.load(open(RES, encoding="utf-8"))
    regions = d["regions"]
    A = line_boxes(FA); B = line_boxes(FB)
    TOL = 8
    all_miss = []; all_fp = []; total_gt = 0
    for p in range(len(A)):
        ta, ba = A[p]; tb, bb = B[p]
        sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
        gt_boxes = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            for idx in range(j1, j2):
                x0, y0, x1, y1 = bb[idx]
                gt_boxes.append((max(0,x0-TOL), max(0,y0-TOL), x1+TOL, y1+TOL, ta[idx] if idx < len(ta) else "<none>", tb[idx]))
        total_gt += len(gt_boxes)
        eng = [r for r in regions if r["page"] == p + 1]
        gt_cov = [False]*len(gt_boxes); reg_used = [False]*len(eng)
        for gi, g in enumerate(gt_boxes):
            for ri, r in enumerate(eng):
                rb = (r["x"], r["y"], r["x"]+r["width"], r["y"]+r["height"])
                if intersect(g[:4], rb) > 0:
                    gt_cov[gi] = True; reg_used[ri] = True
        for gi, g in enumerate(gt_boxes):
            if not gt_cov[gi]:
                all_miss.append((p+1, g))
        for ri, r in enumerate(eng):
            if not reg_used[ri]:
                all_fp.append((p+1, r))
        print(f"page {p+1}: gt_changes={len(gt_boxes)} eng={len(eng)} MISS={sum(1 for x in gt_cov if not x)} FP={sum(1 for x in reg_used if not x)}")

    print(f"\nTotal ground-truth changes: {total_gt}")
    print("\n===== 漏判 =====")
    for pg, g in all_miss:
        print(f"  p{pg} A={g[4]!r}\n       B={g[5]!r}  box=({g[0]},{g[1]})-({g[2]},{g[3]})")
    print("\n===== 误判 =====")
    for pg, r in all_fp:
        print(f"  p{pg} #{r['id']} {r['change_type']}/{r['risk_level']} @({r['x']},{r['y']})+{r['width']}x{r['height']} A={r['template_text']!r}")

if __name__ == "__main__":
    main()
