"""Run the local Pixel Diff FastAPI service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Pixel Diff HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    from pixel_diff._app_paths import is_frozen

    if is_frozen():
        # 冻结态：包已随 exe 捆绑，直接传入 app 对象，不再依赖 src/ 路径
        from pixel_diff_api.app import app

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=False,
        )
    else:
        uvicorn.run(
            "pixel_diff_api.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            app_dir=str(SRC_ROOT),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
