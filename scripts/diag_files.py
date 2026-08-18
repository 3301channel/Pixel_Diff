"""General diagnostic: run full engine on any two files and dump regions + annotations.

Usage:
    python scripts/diag_files.py <template> <scan> [start_page] [end_page] [config.yaml]
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import fitz

from pixel_diff.models import PixelDiffConfig
from pixel_diff.engine import PixelDiffEngine

DEFAULT_CFG = "configs/sensitive_recall_trial.yaml"


def main() -> None:
    tpl = sys.argv[1]
    scan = sys.argv[2]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end = int(sys.argv[4]) if len(sys.argv) > 4 else 999
    cfg_path = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_CFG

    cfg = PixelDiffConfig.from_yaml(cfg_path)
    engine = PixelDiffEngine(cfg)

    n = fitz.open(tpl).page_count
    end = min(end, n - 1)
    print(f"=== TPL={Path(tpl).name} SCAN={Path(scan).name} pages={n} "
          f"config={cfg_path} range={start}-{end} ===")

    out = {}
    for p in range(start, end + 1):
        t0 = time.time()
        try:
            res = engine.compare(scan, tpl, page=p)
            dt = time.time() - t0
            regs = res.differences
            print(f"\n--- page {p} ({dt:.1f}s) regions={len(regs)} ---")
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
            for a in ann:
                print(f"     ann#{a.get('region_id')} T={a.get('template_text')!r} "
                      f"S={a.get('scan_text')!r} identical={a.get('is_identical')}")
            out[str(p)] = dict(regions=rows, annotations=ann)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"\n!!! page {p} EXCEPTION: {e}")
            traceback.print_exc()
            out[str(p)] = dict(error=str(e))

    Path("scripts/diag_files.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== wrote scripts/diag_files.json ===")


if __name__ == "__main__":
    main()
