"""Upload documents to the Pixel-Diff API and download comparison results."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4


class ApiClientError(RuntimeError):
    """Raised when the API cannot complete the requested comparison."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for one document comparison."""

    parser = argparse.ArgumentParser(
        description="上传模板和待检测文档，等待 Pixel-Diff API 完成并下载结果。",
    )
    parser.add_argument("template", type=Path, help="模板文档路径")
    parser.add_argument("candidate", type=Path, help="待检测文档路径")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API 根地址（默认：http://127.0.0.1:8000）",
    )
    parser.add_argument(
        "--config",
        dest="config_name",
        default="sensitive_recall_trial",
        help="服务端配置名称（默认：sensitive_recall_trial）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="跳过相似度预检，低相似度也强制比对",
    )
    parser.add_argument("--output-dir", type=Path, help="结果保存目录")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="状态轮询间隔秒数（默认：1）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="任务等待上限秒数（默认：900）",
    )
    parser.add_argument(
        "--download-reports",
        action="store_true",
        help="同时下载 HTML 和 DOCX 报告",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="不下载每页对比图",
    )
    return parser.parse_args(argv)


def build_multipart_body(
    fields: Mapping[str, str],
    files: Mapping[str, Path],
) -> tuple[bytes, str]:
    """Encode form fields and local files as multipart/form-data."""

    boundary = f"----PixelDiffBoundary{uuid4().hex}"
    marker = boundary.encode("ascii")
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                b"--" + marker + b"\r\n",
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    for name, path in files.items():
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        safe_filename = path.name.replace('"', "_")
        chunks.extend(
            [
                b"--" + marker + b"\r\n",
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{safe_filename}"\r\n'
                ).encode(),
                f"Content-Type: {media_type}\r\n\r\n".encode("ascii"),
                path.read_bytes(),
                b"\r\n",
            ]
        )

    chunks.extend([b"--" + marker + b"--\r\n"])
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def request_json(request: str | Request, timeout: float = 60) -> dict[str, Any]:
    """Send an HTTP request and decode a JSON object response."""

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ApiClientError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ApiClientError(f"cannot connect to API: {exc.reason}") from exc

    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiClientError("API returned invalid JSON") from exc
    if not isinstance(decoded, dict):
        raise ApiClientError("API returned a non-object JSON response")
    return decoded


def _extract_data(payload: dict[str, Any]) -> dict[str, Any]:
    """从 {"code","msg","data"} 响应中取出 data 字典。"""
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


def create_task(
    base_url: str,
    template: Path,
    candidate: Path,
    config_name: str,
    force: bool = False,
    request_json_fn: Callable[[str | Request], dict[str, Any]] = request_json,
) -> dict[str, Any]:
    """Upload the input documents and create one asynchronous task."""

    body, content_type = build_multipart_body(
        {"config_name": config_name, "force": "true" if force else "false"},
        {
            "file_a": template,
            "file_b": candidate,
        },
    )
    request = Request(
        f"{base_url.rstrip('/')}/api/v1/compare/tasks",
        data=body,
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "Accept": "application/json",
        },
        method="POST",
    )
    payload = request_json_fn(request)
    data = payload.get("data") or {}
    task_id = data.get("task_id") if isinstance(data, dict) else None
    if not isinstance(task_id, str) or not task_id:
        raise ApiClientError("create-task response does not contain task_id (expected in data.task_id)")
    return payload


def wait_for_task(
    base_url: str,
    task_id: str,
    timeout: float,
    poll_interval: float,
    request_json_fn: Callable[[str], dict[str, Any]] = request_json,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll an asynchronous comparison task until it completes."""

    started = monotonic_fn()
    task_url = f"{base_url.rstrip('/')}/api/v1/compare/tasks/{task_id}/result"
    while True:
        payload = request_json_fn(task_url)
        data = _extract_data(payload)
        status = data.get("status")
        if status == "completed":
            return payload
        if status == "failed":
            detail = data.get("error") or "unknown service error"
            raise ApiClientError(f"comparison failed: {detail}")
        if status not in {"pending", "processing", "running"}:
            raise ApiClientError(f"API returned an invalid task status: {status!r}")
        if monotonic_fn() - started > timeout:
            raise ApiClientError(f"comparison timed out after {timeout:.1f} seconds")
        sleep_fn(poll_interval)


def download_file(url: str, destination: Path, timeout: float = 60) -> None:
    """Download a binary response to a local file."""

    try:
        with urlopen(url, timeout=timeout) as response:
            content = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ApiClientError(f"HTTP {exc.code} while downloading {url}: {detail}") from exc
    except URLError as exc:
        raise ApiClientError(f"cannot download {url}: {exc.reason}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def _validate_args(args: argparse.Namespace) -> None:
    for path in (args.template, args.candidate):
        if not path.is_file():
            raise ApiClientError(f"input file does not exist: {path}")
    if args.poll_interval < 0:
        raise ApiClientError("--poll-interval must be non-negative")
    if args.timeout <= 0:
        raise ApiClientError("--timeout must be greater than zero")


def _download_optional(
    url: str,
    destination: Path,
    label: str,
) -> bool:
    try:
        download_file(url, destination)
    except ApiClientError as exc:
        print(f"warning: unable to download {label}: {exc}", file=sys.stderr)
        return False
    print(f"downloaded {label}: {destination}")
    return True


def _download_page_images(base_url: str, task_id: str, output_dir: Path, max_pages: int = 200) -> int:
    """按页探测并下载差异热力图（diff），直到服务端返回 404 为止。

    服务端结果 JSON 不暴露总页数，因此采用"探测到 404 即结束"的策略。
    正确的图片路由为 /api/pixel/compare/tasks/{task_id}/pages/{page}/diff。
    """
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for page in range(1, max_pages + 1):
        url = f"{base_url.rstrip('/')}/api/pixel/compare/tasks/{task_id}/pages/{page}/diff"
        dest = images_dir / f"page_{page:04d}_heatmap.png"
        try:
            download_file(url, dest)
        except ApiClientError as exc:
            if "HTTP 404" in str(exc):
                break
            print(f"warning: stopped page-image download at page {page}: {exc}", file=sys.stderr)
            break
        print(f"downloaded page {page} image: {dest}")
        downloaded += 1
    if downloaded == 0:
        print("warning: no page images downloaded", file=sys.stderr)
    return downloaded


def run(args: argparse.Namespace) -> Path:
    """Execute one full API comparison and return the output directory."""

    _validate_args(args)
    base_url = args.base_url.rstrip("/")
    created = create_task(
        base_url,
        args.template,
        args.candidate,
        args.config_name,
        force=args.force,
    )
    task_id = str(_extract_data(created)["task_id"])
    print(f"task created: {task_id}")

    completed = wait_for_task(
        base_url,
        task_id,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
    )
    completed_data = _extract_data(completed)
    difference_count = completed_data.get("difference_count")
    total_pages = completed_data.get("total_pages")
    print(
        f"task completed: pages={total_pages if total_pages is not None else '?'} "
        f"differences={difference_count if difference_count is not None else '?'}"
    )

    output_dir = args.output_dir or Path("artifacts") / "api_client" / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    result_url = f"{base_url}/api/v1/compare/tasks/{task_id}/result"
    result = request_json(result_url)
    result_path = output_dir / "diff_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"downloaded JSON result: {result_path}")

    if not args.no_images:
        _download_page_images(base_url, task_id, output_dir)

    if args.download_reports:
        for report_type in ("html", "docx"):
            _download_optional(
                f"{base_url}/api/v1/compare/tasks/{task_id}/report/{report_type}",
                output_dir / f"diff_report.{report_type}",
                f"{report_type.upper()} report",
            )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    try:
        output_dir = run(args)
    except (ApiClientError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"all available outputs saved to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
