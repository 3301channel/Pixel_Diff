"""FastAPI application — document comparison service (4 endpoints)."""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)

from pixel_diff_api.settings import RUNTIME_SETTINGS_PATH, ApiSettings, RuntimeSettings
from pixel_diff_api.system_info import cpu_count, get_memory_info
from pixel_diff_api.task_service import TaskManager
from pixel_diff_api.viewer import (
    render_compare_viewer,
    render_failed_viewer,
    render_loading_viewer,
    render_settings_page,
)


class _MemoryHandler(logging.Handler):
    """把格式化后的日志行同时保留在内存环形缓冲区，便于通过导出接口直接读取。"""

    def __init__(self, buffer: deque[str]) -> None:
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append(self.format(record))
        except Exception:
            self.handleError(record)


_OK = 200
_PROCESSING = 202


def _ok(data: object, msg: str = "OK") -> dict[str, object]:
    return {"code": _OK, "msg": msg, "data": data}


def create_app(
    settings: ApiSettings | None = None,
    manager: TaskManager | None = None,
) -> FastAPI:
    api_settings = settings or ApiSettings()
    task_manager = manager or TaskManager(api_settings)
    # 运行时设置：真实 TaskManager 自带；注入的测试替身可能没有，兜底新建一个。
    runtime_settings = (
        getattr(task_manager, "runtime_settings", None)
        or RuntimeSettings(RUNTIME_SETTINGS_PATH)
    )
    application = FastAPI(title="文档对比服务", version="1.0.0", docs_url=None)

    # ── 日志：写文件（artifacts/api_logs/api_YYYYMMDD.log）+ 内存环形缓冲（最近 500 条）──
    logger = logging.getLogger("pixel_diff_api")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_dir = api_settings.project_root / "artifacts" / "api_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"api_{datetime.now(UTC).strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(file_handler)
        memory_buffer: deque[str] = deque(maxlen=500)
        memory_handler = _MemoryHandler(memory_buffer)
        memory_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(memory_handler)
    else:
        memory_buffer = next(
            (h.buffer for h in logger.handlers if isinstance(h, _MemoryHandler)),
            deque(maxlen=500),
        )
    application.state.api_logger = logger
    application.state.api_log_buffer = memory_buffer

    @application.middleware("http")
    async def log_http_request(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception("request failed: %s %s -> %s", request.method, request.url.path, exc)
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        query = f"?{request.url.query}" if request.url.query else ""
        logger.info(
            "http %s %s%s -> %d (%.1fms) client=%s bytes=%s",
            request.method,
            request.url.path,
            query,
            response.status_code,
            elapsed_ms,
            request.client.host if request.client else "-",
            response.headers.get("content-length", "-"),
        )
        return response

    @application.get("/docs", include_in_schema=False)
    def swagger_ui() -> HTMLResponse:
        return _chinese_swagger(application)

    # ── 1. 创建比对任务 ──────────────────────────────────────────
    @application.post(
        "/api/v1/compare/tasks",
        status_code=status.HTTP_202_ACCEPTED,
        tags=["文档对比"],
        summary="上传文件并创建比对任务",
    )
    async def create_compare_task(
        request: Request,
        file_a: Annotated[UploadFile, File(description="原件 A")],
        file_b: Annotated[UploadFile, File(description="比对稿 B")],
        config_name: Annotated[str, Form()] = "sensitive_recall_trial",
        ignore_categories: Annotated[str, Form()] = "[]",
        space_recognition: Annotated[bool, Form()] = False,
        force: Annotated[
            bool, Form(description="true 时跳过相似度预检，低相似度也强制比对")
        ] = False,
    ) -> JSONResponse:
        try:
            suffix_a = api_settings.validate_extension(file_a.filename or "")
            suffix_b = api_settings.validate_extension(file_b.filename or "")
        except ValueError as exc:
            return JSONResponse(
                {"code": 40001, "msg": str(exc), "data": None},
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        task_id = uuid4().hex
        logger.info(
            "create_task begin task_id=%s file_a=%s file_b=%s",
            task_id, file_a.filename, file_b.filename,
        )
        input_dir = api_settings.task_root / task_id / "inputs"
        input_dir.mkdir(parents=True, exist_ok=False)
        path_a = input_dir / f"file_a{suffix_a}"
        path_b = input_dir / f"file_b{suffix_b}"
        try:
            await _save_upload(file_a, path_a, api_settings.max_upload_bytes)
            await _save_upload(file_b, path_b, api_settings.max_upload_bytes)
            logger.info(
                "create_task upload saved task_id=%s file_a=%s(%dB) file_b=%s(%dB) "
                "config_name=%s ignore_categories=%s space_recognition=%s",
                task_id, path_a.name, path_a.stat().st_size,
                path_b.name, path_b.stat().st_size,
                config_name, ignore_categories, space_recognition,
            )
        except ValueError as exc:
            shutil.rmtree(api_settings.task_root / task_id, ignore_errors=True)
            return JSONResponse(
                {"code": 40001, "msg": str(exc), "data": None},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        finally:
            await file_a.close()
            await file_b.close()

        task = task_manager.register(
            task_id,
            template_path=path_a,
            candidate_path=path_b,
            config_name=config_name,
            orig_name_a=file_a.filename or "",
            orig_name_b=file_b.filename or "",
            force=force,
        )
        logger.info("create_task submitted task_id=%s status=%s", task_id, task.status)
        info = task.public_dict()
        base = str(request.base_url).rstrip("/")
        info["status_url"] = f"{base}/api/v1/compare/tasks/{task_id}/result"
        info["result_url"] = f"{base}/api/v1/compare/tasks/{task_id}/result"
        logger.info(
            "create_task respond task_id=%s status=%s status_url=%s",
            task_id, task.status, info["status_url"],
        )
        return JSONResponse(
            _ok(info, "Task submitted successfully"),
            status_code=status.HTTP_202_ACCEPTED,
        )

    # ── 2. 查询比对结果 ──────────────────────────────────────────
    @application.get(
        "/api/v1/compare/tasks/{task_id}/result",
        tags=["文档对比"],
        summary="查询任务状态与比对结果",
        description=(
            "轮询任务状态；任务完成时返回完整 JSON 差异结果。"
            "processing 时建议间隔 2～3 秒轮询。"
        ),
    )
    def get_compare_result(task_id: str, request: Request) -> JSONResponse:
        task = task_manager.get(task_id)
        if task is None:
            logger.warning("query result task not found task_id=%s", task_id)
            raise HTTPException(status_code=404, detail="task not found")
        logger.info("query result task_id=%s status=%s", task_id, task.status)

        # 未完成：返回状态信息
        if task.status in ("pending", "running"):
            logger.info(
                "query result respond task_id=%s status=%s results=null",
                task_id, task.status,
            )
            return JSONResponse(
                _ok({**task.public_dict(), "results": None}),
                status_code=_PROCESSING,
            )

        # 失败：返回错误信息
        if task.status == "failed":
            logger.info(
                "query result respond task_id=%s status=failed error=%s",
                task_id, task.error,
            )
            return JSONResponse({
                "code": 50002,
                "msg": task.error or "comparison failed",
                "data": {**task.public_dict(), "results": None},
            })

        # 完成：返回完整结果
        raw = json.loads(Path(task.result_json or "").read_text(encoding="utf-8"))
        results = _build_results(task_id, raw, str(request.base_url).rstrip("/"))
        logger.info(
            "query result respond task_id=%s status=completed regions=%s similarity=%s",
            task_id, len(results.get("diff_list", [])), results.get("similarity"),
        )
        return JSONResponse(_ok({**task.public_dict(), "results": results}))

    # ── 3. 获取可视化页面地址 ────────────────────────────────────
    @application.get(
        "/api/pixel/compare/tasks/{task_id}/view",
        tags=["文档对比展示"],
        summary="获取三栏比对页面地址",
        response_model=None,
    )
    def get_compare_view(task_id: str, request: Request):
        task = task_manager.get(task_id)
        if task is None:
            logger.warning("query view task not found task_id=%s", task_id)
            raise HTTPException(status_code=404, detail="task not found")
        logger.info("query view task_id=%s status=%s", task_id, task.status)
        if task.status == "completed":
            return RedirectResponse(
                f"/api/pixel/compare/tasks/{task_id}/viewer",
                status_code=status.HTTP_302_FOUND,
            )
        if task.status == "failed":
            # 失败/文档不一致：展示两份文件原图 + 顶部错误信息栏
            logger.info("serve view failed task_id=%s error=%s", task_id, task.error)
            return HTMLResponse(
                render_failed_viewer(task, f"/api/pixel/compare/tasks/{task_id}/pages/")
            )
        # 未完成：直接返回 HTML loading 页（页面自动轮询直到完成后展示结果）
        logger.info("serve view loading task_id=%s status=%s", task_id, task.status)
        return HTMLResponse(render_loading_viewer(task))

    # ── 4. 删除任务 ──────────────────────────────────────────────
    @application.delete(
        "/api/v1/compare/tasks/{task_id}",
        tags=["文档对比"],
        summary="删除检测任务及相关数据",
    )
    def delete_compare_task(task_id: str) -> JSONResponse:
        try:
            deleted = task_manager.delete(task_id)
        except RuntimeError as exc:
            logger.warning("delete task rejected task_id=%s reason=%s", task_id, exc)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            logger.warning("delete task not found task_id=%s", task_id)
            raise HTTPException(status_code=404, detail="task not found")
        logger.info("delete task task_id=%s", task_id)
        return JSONResponse(_ok({"task_id": task_id}, "Task files deleted successfully"))

    # ── 5. 导出日志（调试用）────────────────────────────────────
    @application.get(
        "/api/v1/logs/export",
        tags=["调试"],
        summary="导出服务日志",
        description=(
            "导出最近 500 条接口调用日志；可用 task_id 过滤；"
            "format=json 返回结构化日志，默认返回纯文本。"
        ),
        response_model=None,
    )
    def export_logs(
        task_id: str | None = None,
        format: str = "txt",
    ) -> JSONResponse | Response:
        lines = list(application.state.api_log_buffer)
        if task_id:
            lines = [ln for ln in lines if task_id in ln]
        if format == "json":
            records: list[dict[str, str]] = []
            for ln in lines:
                parts = ln.split(" ", 3)
                records.append({
                    "time": f"{parts[0]} {parts[1]}" if len(parts) > 1 else ln,
                    "level": parts[2] if len(parts) > 2 else "",
                    "message": parts[3] if len(parts) > 3 else "",
                })
            return JSONResponse(_ok(records, "logs exported"))
        return Response(
            content=("\n".join(lines) + "\n" if lines else ""),
            media_type="text/plain; charset=utf-8",
        )

    # ── 6. 系统设置页与运行时设置 ────────────────────────────────
    @application.get("/settings", include_in_schema=False)
    def settings_page() -> HTMLResponse:
        """返回系统设置页（核数调整 + 实时内存）。"""
        return HTMLResponse(
            render_settings_page(
                workers=runtime_settings.report_workers,
                cpu_count=cpu_count(),
            )
        )

    @application.get(
        "/api/v1/settings",
        tags=["系统设置"],
        summary="查询当前运行时设置",
    )
    def get_settings() -> JSONResponse:
        max_workers = cpu_count()
        return JSONResponse(_ok({
            "report_workers": runtime_settings.report_workers,
            "cpu_count": max_workers,
            "min": 1,
            "max": max_workers,
        }))

    @application.post(
        "/api/v1/settings",
        tags=["系统设置"],
        summary="确认并生效运行时设置（当前支持核数）",
    )
    def update_settings(payload: dict) -> JSONResponse:
        try:
            workers = int(payload.get("report_workers"))
        except (TypeError, ValueError):
            return JSONResponse(
                {"code": 40001, "msg": "report_workers must be an integer", "data": None},
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        max_workers = cpu_count()
        if workers < 1 or workers > max_workers:
            return JSONResponse(
                {
                    "code": 40001,
                    "msg": f"report_workers must be between 1 and {max_workers}",
                    "data": None,
                },
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        runtime_settings.set_report_workers(workers)
        logger.info("settings update report_workers=%s", workers)
        return JSONResponse(_ok({
            "report_workers": runtime_settings.report_workers,
            "cpu_count": max_workers,
        }))

    @application.get(
        "/api/v1/system/memory",
        tags=["系统设置"],
        summary="实时系统与进程内存占用",
    )
    def get_system_memory() -> JSONResponse:
        return JSONResponse(_ok(get_memory_info()))

    # ── 辅助路由（被可视化页面依赖）───────────────────────────────
    @application.get(
        "/api/pixel/compare/tasks/{task_id}/viewer",
        response_class=HTMLResponse,
        include_in_schema=False,
    )
    def get_compare_viewer(task_id: str) -> HTMLResponse:
        task = task_manager.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        if task.status == "completed":
            if task.result_json is None:
                raise HTTPException(status_code=404, detail="result JSON not found")
            payload = json.loads(Path(task.result_json).read_text(encoding="utf-8"))
            logger.info(
                "serve viewer result task_id=%s pages=%s regions=%s",
                task_id, payload.get("total_pages"), payload.get("total_regions"),
            )
            return HTMLResponse(render_compare_viewer(task, payload))
        if task.status == "failed":
            # 失败/文档不一致：展示两份文件原图 + 顶部错误信息栏
            logger.info("serve viewer failed task_id=%s error=%s", task_id, task.error)
            return HTMLResponse(
                render_failed_viewer(task, f"/api/pixel/compare/tasks/{task_id}/pages/")
            )
        # pending / running → 返回 loading 页（前端轮询直到完成）
        logger.info("serve viewer loading task_id=%s status=%s", task_id, task.status)
        return HTMLResponse(render_loading_viewer(task))

    @application.get(
        "/api/pixel/compare/tasks/{task_id}/pages/{page}/{image_type}",
        include_in_schema=False,
    )
    def get_compare_page_image(task_id: str, page: int, image_type: str) -> FileResponse:
        if page < 1:
            raise HTTPException(status_code=400, detail="page must be at least 1")
        suffixes = {"template": "original", "candidate": "candidate", "diff": "heatmap"}
        if image_type not in suffixes:
            raise HTTPException(
                status_code=400,
                detail="image_type must be template, candidate or diff",
            )
        task = task_manager.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        # failed 时也允许取原图（供"文档不一致"页展示两份文件）
        if task.status not in ("completed", "failed"):
            raise HTTPException(status_code=409, detail=f"task status is {task.status}")
        # 待检原始扫描件是 JPEG（体积小），模板也是 JPEG（避免 PNG 700K~975K 翻页卡），
        # 残影（heatmap）仍是 PNG（差异图简单，PNG 体积小且无有损）。
        ext = "jpg" if image_type in ("template", "candidate") else "png"
        media_type = "image/jpeg" if image_type in ("template", "candidate") else "image/png"
        image_path = (
            Path(task.output_dir or "")
            / "report"
            / f"page_{page:04d}_{suffixes[image_type]}.{ext}"
        )
        if not image_path.is_file():
            raise HTTPException(status_code=404, detail="page image not found")
        return FileResponse(
            image_path,
            media_type=media_type,
            filename=image_path.name,
            content_disposition_type="inline",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @application.get(
        "/api/pixel/compare/tasks/{task_id}/input_image/{kind}",
        include_in_schema=False,
    )
    def get_input_raw(task_id: str, kind: str):
        """返回文件的首张预览图（PDF 渲染第一页，图片直接返回），供展示页使用。"""
        if kind not in {"template", "scan"}:
            raise HTTPException(status_code=400, detail="kind must be template or scan")
        task = task_manager.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        needle = "file_a" if kind == "template" else "file_b"
        input_dir = api_settings.task_root / task_id / "inputs"
        files = sorted(input_dir.glob(f"{needle}.*"))
        if not files:
            raise HTTPException(status_code=404, detail="original input file not found")
        path = files[0]
        if path.suffix.lower() == ".pdf":
            from io import BytesIO

            import cv2

            from pixel_diff.io import load_document_page_bgr
            page_img = load_document_page_bgr(path, page=0, dpi=150)
            _, buf = cv2.imencode(".png", page_img)
            return Response(content=BytesIO(buf).getvalue(), media_type="image/png")
        return FileResponse(path, media_type=f"image/{path.suffix.lstrip('.')}",
                            filename=path.name, content_disposition_type="inline")

    application.state.api_settings = api_settings
    application.state.task_manager = task_manager
    application.state.runtime_settings = runtime_settings

    @application.on_event("shutdown")
    def _shutdown() -> None:
        """应用关闭时回收线程池，避免后台线程泄漏。"""
        try:
            mgr = application.state.task_manager
            if hasattr(mgr, "shutdown"):
                mgr.shutdown(wait=False)
        except Exception:
            logger.exception("shutdown handler failed")

    return application


# ── helpers ───────────────────────────────────────────────────────
def _chinese_swagger(application: FastAPI) -> HTMLResponse:
    openapi_url = application.openapi_url
    if openapi_url is None:
        raise RuntimeError("OpenAPI schema is disabled")
    page = get_swagger_ui_html(
        openapi_url=openapi_url,
        title="文档对比服务接口文档",
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "filter": True,
        },
    )
    html = bytes(page.body).decode("utf-8")
    translations = """<script>
(() => {
  const tr = new Map([
    ["Authorize","认证"],["Try it out","调试"],["Cancel","取消"],
    ["Execute","发送"],["Clear","清空"],["Parameters","请求参数"],
    ["Request body","请求体"],["Responses","响应结果"],["Response body","响应内容"],
    ["Server response","服务器响应"],["No parameters","无请求参数"],
    ["Example Value","示例值"],["Schema","数据结构"],["Models","数据模型"],
    ["Download file","下载文件"]
  ]);
  new MutationObserver((items) => {
    for (const item of items)
      for (const node of item.addedNodes)
        if (node.nodeType === Node.ELEMENT_NODE) {
          const w = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
          const ns = [];
          while (w.nextNode()) ns.push(w.currentNode);
          for (const n of ns) {
            const v = n.nodeValue.trim();
            if (tr.has(v)) n.nodeValue = n.nodeValue.replace(v, tr.get(v));
          }
        }
  }).observe(document.body, {childList: true, subtree: true});
})();
</script>"""
    return HTMLResponse(html.replace("</body>", f"{translations}</body>"))


async def _save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> None:
    size = 0
    with destination.open("wb") as target:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"file exceeds maximum size of {max_bytes} bytes")
            target.write(chunk)


def _map_status(s: str) -> str:
    return {"pending": "pending", "running": "processing"}.get(s, s)


def _build_results(task_id: str, raw: dict, base_url: str = "") -> dict:
    regions = raw.get("regions", [])
    change_counter = Counter(
        r.get("change_type", "modified") for r in regions
    )
    similarity = round(100 * (1 - raw.get("difference_rate", 0)), 1)

    diff_list: list[dict] = []
    for r in regions:
        text = (r.get("template_text") or "").strip()
        diff_list.append({
            "tag": _change_tag(r.get("change_type", "modified")),
            "text_a": text if r.get("change_type") in ("modified", "deleted") else "",
            "text_b": text if r.get("change_type") in ("modified", "added") else "",
            "bbox_a": [{
                "char": "",
                "bbox": [r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]],
                "page_idx": int(r.get("page", 1)) - 1,
            }],
            "bbox_b": None,
            "page_idx": int(r.get("page", 1)) - 1,
            "risk_level": r.get("risk_level"),
            "risk_reason": r.get("risk_reason"),
            "area": r.get("area"),
        })
    return {
        "similarity": similarity,
        "difference_count": {
            "added": change_counter.get("added", 0),
            "deleted": change_counter.get("deleted", 0),
            "modified": change_counter.get("modified", 0),
        },
        "compare_view_url": f"{base_url}/api/pixel/compare/tasks/{task_id}/viewer",
        "preview_url_a": None,
        "preview_url_b": None,
        "diff_list": diff_list,
    }


def _change_tag(ct: str) -> str:
    return {"added": "insert", "deleted": "delete", "modified": "replace"}.get(ct, "replace")


app = create_app()
