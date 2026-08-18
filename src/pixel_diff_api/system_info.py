"""跨平台系统信息读取（无第三方依赖）。

为设置页提供实时内存占用与 CPU 核数：
- 系统内存：总 / 可用 / 已用 / 使用百分比
- 进程内存：当前服务进程的常驻内存（RSS）
- CPU 核数：用于核数设置的合法范围上限

实现策略（仅标准库）：
- Windows：``ctypes`` 调用 ``GlobalMemoryStatusEx`` 与 ``GetProcessMemoryInfo``
- Linux：读取 ``/proc/meminfo`` 与 ``/proc/self/status``
- macOS：``sysconf`` 兜底（总内存可用，可用内存近似）
"""

from __future__ import annotations

import os
import sys
from typing import Any


def cpu_count() -> int:
    """逻辑 CPU 核数，用于核数设置上限；无法探测时回退 1。"""
    return os.cpu_count() or 1


def get_memory_info() -> dict[str, Any]:
    """返回内存信息字典（字节与百分比）。"""
    total, available, used, percent = _system_memory()
    process_rss = _process_rss()
    return {
        "total": int(total),
        "available": int(available),
        "used": int(used),
        "percent": float(percent),
        "process_rss": int(process_rss),
        "cpu_count": cpu_count(),
    }


# ── 系统内存 ────────────────────────────────────────────────────────────────

def _system_memory() -> tuple[int, int, int, float]:
    if sys.platform == "win32":
        return _windows_system_memory()
    if sys.platform == "darwin":
        return _macos_system_memory()
    return _linux_system_memory()


def _linux_system_memory() -> tuple[int, int, int, float]:
    info: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                info[key.strip()] = int(value.strip().split()[0]) * 1024  # kB → bytes
    except OSError:
        return 0, 0, 0, 0.0
    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", info.get("MemFree", 0))
    used = max(0, total - available)
    percent = round(used / total * 100, 1) if total else 0.0
    return total, available, used, percent


def _windows_system_memory() -> tuple[int, int, int, float]:
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    except OSError:
        return 0, 0, 0, 0.0
    total = int(stat.ullTotalPhys)
    available = int(stat.ullAvailPhys)
    used = max(0, total - available)
    percent = float(stat.dwMemoryLoad)
    return total, available, used, percent


def _macos_system_memory() -> tuple[int, int, int, float]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        total = page_size * total_pages
        used_pages = total_pages - os.sysconf("SC_AVPHYS_PAGES")
        used = page_size * used_pages
        available = total - used
        percent = round(used / total * 100, 1) if total else 0.0
        return total, available, used, percent
    except (ValueError, OSError):
        return 0, 0, 0, 0.0


# ── 进程内存 ────────────────────────────────────────────────────────────────

def _process_rss() -> int:
    if sys.platform == "win32":
        return _windows_process_rss()
    if sys.platform == "darwin":
        return 0
    return _linux_process_rss()


def _linux_process_rss() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except OSError:
        pass
    return 0


def _windows_process_rss() -> int:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = kernel32.GetCurrentProcess()
        psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return 0
        return int(counters.WorkingSetSize)
    except OSError:
        return 0
