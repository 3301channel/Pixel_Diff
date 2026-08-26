"""Thread-safe task registry and subprocess-based comparison runner."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from pixel_diff._app_paths import compare_entry, is_frozen
from pixel_diff_api.settings import RUNTIME_SETTINGS_PATH, ApiSettings, RuntimeSettings

TaskStatus = Literal["pending", "running", "completed", "failed"]

logger = logging.getLogger("pixel_diff_api")

# 快速预检：仅在两图特征完全无法配准时才判定为「两份文档几乎完全不同」。
# 注意：不能用原始像素差——像素差在配准前会因尺寸/旋转/偏移不同而被放大，
# 导致「同一文档的不同尺寸/方向版本」被误判为差异过大。改用对齐不变的特征匹配。
_QUICK_CHECK_MAX_DIM = 512  # 降采样长边，加速预检
_QUICK_CHECK_MIN_MATCHES = 25  # 低于此匹配数视为无法配准
_QUICK_CHECK_MIN_INLIER_RATIO = 0.05  # RANSAC 内点率低于此值视为无法配准


def _quick_mismatch_check(candidate_path: Path, template_path: Path) -> bool:
    """对齐不变的快速预检：仅当两图特征完全无法配准时判定为不同文档。

    目的：在全量配准+比对之前快速短路「明显无关」的文件，节省时间。
    策略：对两张灰度图降采样后提取 ORB 特征 → 暴力匹配 → RANSAC 估计单应性。
    ORB 特征匹配对平移/旋转/缩放/光照不敏感，因此「同一文档的扫描版 vs 电子版、
    不同尺寸、轻微旋转」仍能匹配成功，不会被误判；只有两图内容毫不相干时，
    特征几乎无对应点或内点率极低，才判定为不同文档。
    """
    tmpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    cand = cv2.imread(str(candidate_path), cv2.IMREAD_GRAYSCALE)
    if tmpl is None or cand is None:
        # 非图像（如 PDF）→ 跳过预检，交给完整配准流程判定
        return False

    def _downscale(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) <= _QUICK_CHECK_MAX_DIM:
            return img
        scale = _QUICK_CHECK_MAX_DIM / max(h, w)
        return cv2.resize(
            img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    tmpl_s = _downscale(tmpl)
    cand_s = _downscale(cand)

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(tmpl_s, None)
    kp2, des2 = orb.detectAndCompute(cand_s, None)
    if des1 is None or des2 is None or len(des1) < 20 or len(des2) < 20:
        return False  # 特征不足无法判定，交给完整流程

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    try:
        matches = matcher.match(des1, des2)
    except cv2.error:
        return False
    if len(matches) < _QUICK_CHECK_MIN_MATCHES:
        return True  # 几乎无对应点 → 不同文档

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if homography is None:
        return True  # 无法估计单应性 → 配准不可能 → 不同文档
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = inliers / max(1, len(matches))
    return inlier_ratio < _QUICK_CHECK_MIN_INLIER_RATIO


def _extract_document_mismatch_message(stderr: str) -> str:
    """从 stderr 里提取「文档不一致」中文文案行（含「疑似非同一文档」的那一行）。

    配准失败（AlignmentError）时 compare.py 会把统一的中文文案打印到 stderr，
    但日志（INFO compare start …）也混在 stderr 里。此函数只取中文文案那一行，
    避免把日志一起塞进 task.error，保证「特征匹配失败」与「配准后判定失败」
    两条路径返回同样干净的中文「文档不一致」提示。
    """
    for line in (stderr or "").splitlines():
        stripped = line.strip()
        if "疑似非同一文档" in stripped or "无法配准" in stripped:
            return stripped
    return ""


# 内存缓存 TTL：已完成/失败的任务在内存中保留 1 小时后自动移除
_STALE_TASK_TTL = timedelta(hours=1)


def _map_status(s: TaskStatus) -> str:
    """映射内部状态到文档对外状态。"""
    return {"pending": "pending", "running": "processing"}.get(s, s)


@dataclass
class CompareTask:
    task_id: str
    status: TaskStatus = "pending"
    file_name_a: str = ""
    file_name_b: str = ""
    config_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    completed_at: str | None = None  # alias: 文档用 completed_at，内部保留 finished_at 也用
    output_dir: str | None = None
    result_json: str | None = None
    difference_count: int | None = None
    total_pages: int | None = None
    error: str | None = None

    def public_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "status": _map_status(self.status),
            "queued_ahead": 0,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.finished_at,
            "error": self.error,
            "file_name_a": self.file_name_a,
            "file_name_b": self.file_name_b,
        }


class TaskManager:
    def __init__(
        self,
        settings: ApiSettings,
        runtime_settings: RuntimeSettings | None = None,
    ) -> None:
        self.settings = settings
        self.runtime_settings = runtime_settings or RuntimeSettings(RUNTIME_SETTINGS_PATH)
        self._tasks: dict[str, CompareTask] = {}
        self._futures: dict[str, Future[None]] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_tasks,
            thread_name_prefix="pixel-diff-api",
        )
        self._cleanup_timer: threading.Timer | None = None
        self._start_cleanup_timer()

    def register(
        self,
        task_id: str,
        template_path: Path,
        candidate_path: Path,
        config_name: str,
        orig_name_a: str = "",
        orig_name_b: str = "",
        force: bool = False,
    ) -> CompareTask:
        task = CompareTask(
            task_id=task_id,
            file_name_a=orig_name_a or template_path.name,
            file_name_b=orig_name_b or candidate_path.name,
            config_name=config_name,
        )
        with self._lock:
            self._tasks[task_id] = task
            self._futures[task_id] = self._executor.submit(
                self._run_task,
                task_id,
                template_path,
                candidate_path,
                config_name,
                force,
            )
        logger.info(
            "task register task_id=%s config=%s template=%s candidate=%s",
            task_id, config_name, template_path.name, candidate_path.name,
        )
        return task

    def get(self, task_id: str) -> CompareTask | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                return task
        # 内存没有 → 尝试从磁盘恢复（服务重启后历史任务仍可查询）
        return self._restore_from_disk(task_id)

    def _restore_from_disk(self, task_id: str) -> CompareTask | None:
        """从 artifacts/api_tasks/{task_id}/ 磁盘目录恢复已完成任务。

        服务重启后 TaskManager 内存清空，但已完成任务的输入文件与
        diff_result.json 仍保留在磁盘。据此重建 CompareTask，返回
        completed 状态；磁盘数据不完整时返回 None。
        """
        task_dir = self.settings.task_root / task_id
        if not task_dir.is_dir():
            return None
        input_dir = task_dir / "inputs"
        outputs_dir = task_dir / "outputs"
        result_files = sorted(outputs_dir.rglob("diff_result.json")) if outputs_dir.is_dir() else []
        if not result_files:
            return None
        result_path = result_files[0]
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        def _input_name(primary_key: str, *glob_patterns: str) -> str:
            """恢复输入文件名：优先 diff_result 里的路径，回退 inputs 目录（兼容两种命名）。"""
            name = Path(str(payload.get(primary_key, ""))).name
            if name:
                return name
            if input_dir.is_dir():
                for pattern in glob_patterns:
                    hit = next(input_dir.glob(pattern), None)
                    if hit is not None:
                        return hit.name
            return ""

        file_name_a = _input_name("template_path", "template*", "file_a*")
        file_name_b = _input_name("scan_path", "candidate*", "file_b*")
        try:
            mtime = datetime.fromtimestamp(result_path.stat().st_mtime, tz=UTC).isoformat()
        except OSError:
            mtime = datetime.now(UTC).isoformat()
        task = CompareTask(
            task_id=task_id,
            status="completed",
            file_name_a=file_name_a,
            file_name_b=file_name_b,
            config_name="",
            created_at=str(payload.get("run_id", "")),
            started_at=None,
            finished_at=mtime,
            completed_at=mtime,
            output_dir=str(result_path.parent),
            result_json=str(result_path),
            difference_count=int(payload.get("total_regions", 0)),
            total_pages=int(payload.get("total_pages", 0)),
        )
        with self._lock:
            self._tasks[task_id] = task
        logger.info(
            "task restored from disk task_id=%s regions=%s output=%s",
            task_id, task.difference_count, task.output_dir,
        )
        return task

    def delete(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status in {"pending", "running"}:
                raise RuntimeError("running tasks cannot be deleted")
            self._tasks.pop(task_id, None)
            self._futures.pop(task_id, None)
        task_dir = (self.settings.task_root / task_id).resolve()
        root = self.settings.task_root.resolve()
        if task_dir.parent != root:
            raise RuntimeError("invalid task directory")
        shutil.rmtree(task_dir, ignore_errors=True)
        logger.info("task deleted task_id=%s dir_removed=%s", task_id, task_dir)
        return True

    def shutdown(self, wait: bool = False) -> None:
        """关闭线程池并清理所有内存缓存。

        应在应用退出前调用，确保线程资源正常回收。
        ``wait=True`` 会等待所有运行中任务完成（优雅关闭）。
        """
        logger.info("task manager shutdown begin wait=%s", wait)
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None
        self._executor.shutdown(wait=wait, cancel_futures=not wait)
        with self._lock:
            self._tasks.clear()
            self._futures.clear()
        logger.info("task manager shutdown complete")

    def _start_cleanup_timer(self) -> None:
        """启动定时清理线程：每 30 分钟移除 1 小时前完成的旧任务内存缓存。"""
        self._cleanup_loop()

    def _cleanup_loop(self) -> None:
        self._evict_stale_tasks()
        with self._lock:
            self._cleanup_timer = threading.Timer(1800, self._cleanup_loop)
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()

    def _evict_stale_tasks(self) -> None:
        """从内存缓存移除已完成/失败超过 TTL 的任务（磁盘数据保留）。"""
        cutoff = datetime.now(UTC) - _STALE_TASK_TTL
        with self._lock:
            stale = [
                tid
                for tid, t in self._tasks.items()
                if t.status in ("completed", "failed")
                and t.finished_at is not None
                and datetime.fromisoformat(t.finished_at) < cutoff
            ]
        for tid in stale:
            with self._lock:
                self._tasks.pop(tid, None)
        if stale:
            logger.info("evicted %d stale tasks from memory cache", len(stale))


    def _run_task(
        self,
        task_id: str,
        template_path: Path,
        candidate_path: Path,
        config_name: str,
        force: bool = False,
    ) -> None:
        task = self._require(task_id)
        task.status = "running"
        task.started_at = datetime.now(UTC).isoformat()
        logger.info(
            "task run begin task_id=%s template=%s candidate=%s config=%s force=%s",
            task_id, template_path.name, candidate_path.name, config_name, force,
        )

        # 快速预检：两份文件几乎完全不同 → 直接短路，不跑配准和比对（force=True 跳过）
        if not force and _quick_mismatch_check(candidate_path, template_path):
            task.status = "failed"
            task.error = "两份文档差异过大，疑似非同一文档"
            task.finished_at = datetime.now(UTC).isoformat()
            logger.warning(
                "task run quick mismatch task_id=%s files=(%s, %s)",
                task_id, template_path.name, candidate_path.name,
            )
            return

        task_dir = self.settings.task_root / task_id
        report_root = task_dir / "outputs"
        config_path = self.settings.validate_config_name(config_name)
        entry = compare_entry()
        # 冻结态 entry 本身是可执行文件（compare.exe），无需再套 sys.executable；
        # 开发态 entry 是 scripts/compare.py，需用当前解释器运行。
        command = (
            [str(entry)]
            if is_frozen()
            else [sys.executable, str(entry)]
        ) + [
            str(candidate_path),
            str(template_path),
            "--config",
            str(config_path),
            "--report-dir",
            str(report_root),
            "--run-id",
            task_id,
            "--report-workers",
            str(self.runtime_settings.report_workers),
        ]
        logger.info("task run command task_id=%s argv=%s", task_id, command)
        try:
            completed = subprocess.run(
                command,
                cwd=self.settings.project_root,
                capture_output=True,
                text=True,
                timeout=self.settings.task_timeout_seconds,
                check=False,
            )
            logger.info(
                "task run process task_id=%s returncode=%s stdout_bytes=%s stderr_bytes=%s",
                task_id, completed.returncode,
                len(completed.stdout or ""), len(completed.stderr or ""),
            )
            if completed.returncode != 0:
                message = _extract_document_mismatch_message(completed.stderr or "")
                if not message:
                    message = (
                        (completed.stderr or "").strip()
                        or (completed.stdout or "").strip()
                    )
                raise RuntimeError(message[-4000:] or "comparison process failed")
            result_files = sorted(report_root.rglob("diff_result.json"))
            if len(result_files) != 1:
                raise RuntimeError("comparison did not create exactly one result JSON")
            result_path = result_files[0]
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            task.output_dir = str(result_path.parent)
            task.result_json = str(result_path)
            task.difference_count = int(payload.get("total_regions", 0))
            task.total_pages = int(payload.get("total_pages", 0))
            # 报告模式下指标在 pages[0]，单页模式在顶层 metrics
            pages = payload.get("pages") or []
            page0 = pages[0] if isinstance(pages, list) and pages else {}
            inlier_ratio = float(
                page0.get("inlier_ratio")
                or (payload.get("metrics") or {}).get("inlier_ratio")
                or 1.0
            )
            # local_warp 关闭时 local_warp_gate_foreground_iou 从未计算、恒为 0.0，
            # 不能作为"对齐后前景重合度"信号；仅当 local_warp 实际开启时才参与判定。
            local_warp_enabled = int(page0.get("local_warp_enabled", 0) or 0)
            warp_iou = float(page0.get("local_warp_gate_foreground_iou", 1.0))
            if not local_warp_enabled:
                warp_iou = 1.0
            # 配准是否把图像拉歪/配准失败（顶层由 report 汇总，兜底再逐页判断）
            alignment_distorted = bool(payload.get("alignment_distorted")) or any(
                bool(p.get("alignment_distorted")) for p in pages
            )

            # 仅当「配准把图拉歪 / 配准失败」时才判定为差异过大，避免把"配准乱掉"
            # 的灾难视图渲染给用户。纯内容差异（改字、加行）且配准干净时不触发。
            #  条件 1：配准变形超标（剪切/拉伸/旋转）→ 配准把图拉歪
            #  条件 2：RANSAC 内点率 < 5% → 特征匹配失败、配准本质失败
            #  条件 3：局配形变 IoU < 5% → 对齐后前景完全不重合
            # force=True 时跳过短路，低配准质量也强制生成可视化结果
            if (
                not force
                and (alignment_distorted or inlier_ratio < 0.05 or warp_iou < 0.05)
            ):
                task.status = "failed"
                if alignment_distorted:
                    reason = (
                        "两份文档配准异常（配准将图像拉斜/拉伸变形，"
                        "剪切或缩放超阈值），疑似非同一文档"
                    )
                elif warp_iou < 0.05:
                    reason = "两份文档配准失败（对齐后前景重合度极低），疑似非同一文档"
                else:
                    reason = f"两份文档特征匹配失败（内点率 {inlier_ratio:.1%}），疑似非同一文档"
                task.error = f"{reason}，未生成可视化比对结果"
                logger.warning(
                    "task run document mismatch task_id=%s inlier=%.4f "
                    "warp_iou=%.4f distorted=%s",
                    task_id, inlier_ratio, warp_iou, alignment_distorted,
                )
            else:
                task.status = "completed"
                logger.info(
                    "task run complete task_id=%s pages=%s regions=%s output=%s",
                    task_id, task.total_pages, task.difference_count, task.output_dir,
                )
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
            logger.exception("task run failed task_id=%s reason=%s", task_id, exc)
        finally:
            task.finished_at = datetime.now(UTC).isoformat()
            # 释放 Future 引用，避免 completed/failed 线程对象长期驻留内存
            with self._lock:
                self._futures.pop(task_id, None)

    def _require(self, task_id: str) -> CompareTask:
        with self._lock:
            task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task
