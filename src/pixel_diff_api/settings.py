"""Configuration for the local comparison API."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from pixel_diff._app_paths import app_root

PROJECT_ROOT = app_root()

RUNTIME_SETTINGS_PATH = PROJECT_ROOT / "artifacts" / "runtime_settings.json"


@dataclass(frozen=True)
class ApiSettings:
    project_root: Path = PROJECT_ROOT
    task_root: Path = PROJECT_ROOT / "artifacts" / "api_tasks"
    max_upload_bytes: int = 100 * 1024 * 1024
    max_concurrent_tasks: int = 1
    task_timeout_seconds: int = 900
    allowed_extensions: frozenset[str] = frozenset({".pdf", ".docx", ".png", ".jpg", ".jpeg"})
    config_profiles: dict[str, Path] = field(
        default_factory=lambda: {
            "default": PROJECT_ROOT / "configs" / "default.yaml",
            "sensitive_recall_trial": PROJECT_ROOT
            / "configs"
            / "sensitive_recall_trial.yaml",
        }
    )

    def validate_config_name(self, name: str) -> Path:
        try:
            path = self.config_profiles[name]
        except KeyError as exc:
            allowed = ", ".join(sorted(self.config_profiles))
            raise ValueError(f"unknown config_name; allowed values: {allowed}") from exc
        if not path.is_file():
            raise ValueError(f"configured profile does not exist: {path.name}")
        return path

    def validate_extension(self, filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.allowed_extensions:
            allowed = ", ".join(sorted(self.allowed_extensions))
            raise ValueError(f"unsupported file type; allowed extensions: {allowed}")
        return suffix


class RuntimeSettings:
    """运行时可变设置（当前仅核数），带 JSON 持久化与线程安全读写。

    核数 ``report_workers`` 通过设置页确认后生效：立即写入内存供后续比对
    任务使用，同时落盘到 ``artifacts/runtime_settings.json``，服务重启后仍保留。
    """

    def __init__(self, path: Path, default_workers: int = 8) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._report_workers = self._load_workers(default_workers)

    def _load_workers(self, default_workers: int) -> int:
        if not self._path.is_file():
            return default_workers
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            workers = int(data.get("report_workers", default_workers))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return default_workers
        return max(1, workers)

    @property
    def report_workers(self) -> int:
        with self._lock:
            return self._report_workers

    def set_report_workers(self, workers: int) -> None:
        workers = max(1, int(workers))
        with self._lock:
            self._report_workers = workers
        self._save()

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps({"report_workers": self._report_workers}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            # 持久化失败不阻断设置生效（仅内存态生效）
            pass
