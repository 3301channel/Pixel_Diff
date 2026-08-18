"""详细日志系统初始化。

为比对管道（CLI / 引擎 / Python API）提供统一的详细日志输出。

日志去向：
- 控制台（可选）：默认 INFO 级别，避免刷屏，便于调试时实时观察。
- 文件：默认 DEBUG 级别（最详细），优先写到单次运行目录下的 ``run.log``；
  未指定时回退到全局汇总日志 ``artifacts/logs/pixel_diff_YYYYMMDD.log``。

所有 ``pixel_diff.*`` 子模块（``engine`` / ``io`` / ``alignment`` ...）通过
logging 的命名空间层级自动汇聚到 ``pixel_diff`` logger 上，无需逐模块配置。
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from pixel_diff._app_paths import app_root

LOGGER_NAME = "pixel_diff"

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
_FILE_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 合法日志级别 → 数值，用于校验与解析。
_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def get_logger() -> logging.Logger:
    """返回 ``pixel_diff`` 命名空间的根 logger。"""
    return logging.getLogger(LOGGER_NAME)


def default_log_file() -> Path:
    """全局汇总日志路径：``artifacts/logs/pixel_diff_YYYYMMDD.log``。"""
    log_dir = app_root() / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"pixel_diff_{datetime.now().strftime('%Y%m%d')}.log"


def _level_of(level: str) -> int:
    """将日志级别名解析为数值，非法值回退到 INFO。"""
    return _LEVELS.get(level.upper(), logging.INFO)


def _has_console_handler(logger: logging.Logger) -> bool:
    return any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )


def _has_file_handler(logger: logging.Logger, log_file: Path) -> bool:
    target = str(log_file.resolve())
    return any(
        isinstance(h, logging.FileHandler) and str(Path(h.baseFilename).resolve()) == target
        for h in logger.handlers
    )


def setup_logging(
    log_file: str | Path | None = None,
    level: str = "INFO",
    console: bool = True,
    *,
    file_level: str = "DEBUG",
    enable_file: bool = True,
) -> logging.Logger:
    """初始化 ``pixel_diff`` 日志（幂等，可重复调用追加不同日志文件）。

    Args:
        log_file: 单次运行日志文件路径；``None`` 时回退到全局汇总日志。
        level: 控制台日志级别（默认 ``INFO``）。
        console: 是否输出到控制台（stderr）。
        file_level: 文件日志级别（默认 ``DEBUG``，最详细）。
        enable_file: 是否写入日志文件（默认 ``True``）。

    Returns:
        配置完成的 ``pixel_diff`` logger。
    """
    logger = get_logger()
    # logger 级别取「最详细」的一档，让 DEBUG 消息能到达 file handler；
    # 控制台/file 各自再用 handler 级别过滤，保证终端不刷屏、文件最详细。
    logger.setLevel(min(_level_of(level), _level_of(file_level)))
    logger.propagate = False

    if console and not _has_console_handler(logger):
        handler = logging.StreamHandler()
        handler.setLevel(_level_of(level))
        handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)

    if enable_file:
        if log_file is None:
            log_file = default_log_file()
        log_file = Path(log_file)
        if not _has_file_handler(logger, log_file):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setLevel(_level_of(file_level))
            handler.setFormatter(logging.Formatter(_FILE_FORMAT, _DATE_FORMAT))
            logger.addHandler(handler)

    return logger
