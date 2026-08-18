"""部署到打包机（麒麟）并 PyInstaller 打包。

约定：每次打包放到独立新目录 /home/LENOVO/Pixel_Diff_<时间戳>/，避免新旧代码混淆。
复用旧项目 .venv（软链接），仅上传源码 + spec，打包产物 dist/pd_pyinstaller_build/。

用法：
    .venv/Scripts/python.exe deploy_build.py [sync|build|all]
"""
from __future__ import annotations

import datetime
import os
import posixpath
import sys
from pathlib import Path

import paramiko

# 部署凭据从环境变量读取：PD_KYLIN_HOST / PD_KYLIN_USER(默认 LENOVO) / PD_KYLIN_PWD
HOST = os.environ.get("PD_KYLIN_HOST", "")
USER = os.environ.get("PD_KYLIN_USER", "LENOVO")
PWD = os.environ.get("PD_KYLIN_PWD", "")
REMOTE_BASE = "/home/LENOVO"
VENV_SRC = "/home/LENOVO/Pixel_Diff/.venv"
RUN_DIR_NAME = "Pixel_Diff_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
REMOTE_PROJ = posixpath.join(REMOTE_BASE, RUN_DIR_NAME)

LOCAL_ROOT = Path(__file__).resolve().parent
SYNC_SUBDIRS = ["src", "scripts", "configs"]
SYNC_FILES = ["build_linux_kylin.spec", "pyproject.toml"]
EXCLUDE_PARTS = {"__pycache__", ".venv", "build", "dist", ".egg-info"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def connect() -> paramiko.SSHClient:
    if not HOST or not PWD:
        raise SystemExit(
            "缺少部署凭据：请先设置环境变量 PD_KYLIN_HOST / PD_KYLIN_PWD"
            "（USER 默认 LENOVO，可用 PD_KYLIN_USER 覆盖）"
        )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, 22, USER, PWD, timeout=20)
    return c


def run(c: paramiko.SSHClient, cmd: str, timeout: int | None = None) -> tuple[str, str]:
    _i, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(), e.read().decode()


def collect_files() -> list[Path]:
    files: list[Path] = []
    for sub in SYNC_SUBDIRS:
        base = LOCAL_ROOT / sub
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_PARTS for part in p.parts):
                continue
            if p.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            if ".build" in p.name:
                continue
            files.append(p)
    for name in SYNC_FILES:
        files.append(LOCAL_ROOT / name)
    return files


def do_sync() -> str:
    c = connect()
    sftp = c.open_sftp()
    print(f"[sync] 目标目录: {REMOTE_PROJ}")
    run(c, f"mkdir -p {REMOTE_PROJ}")
    files = collect_files()
    print(f"[sync] 共 {len(files)} 个文件")
    for p in files:
        rel = p.relative_to(LOCAL_ROOT).as_posix()
        remote = posixpath.join(REMOTE_PROJ, rel)
        remote_dir = posixpath.dirname(remote)
        run(c, f"mkdir -p {remote_dir}")
        sftp.put(str(p), remote)
    sftp.close()
    # 软链接 .venv，复用旧项目依赖
    out, err = run(c, f"ln -s {VENV_SRC} {REMOTE_PROJ}/.venv && echo VENV_LINKED")
    print(out.strip() or err.strip())
    c.close()
    return REMOTE_PROJ


def do_build(remote_proj: str | None = None) -> int:
    remote_proj = remote_proj or _latest_run_dir()
    c = connect()
    build_cmd = (
        f"cd {remote_proj} && export LD_LIBRARY_PATH=/opt/python3.12/lib && "
        ".venv/bin/pyinstaller --clean -y build_linux_kylin.spec"
    )
    print(f"[build] 开始打包（耗时数分钟，耐心等待）...")
    print(f"[build] 命令: {build_cmd}")
    out, err = run(c, build_cmd, timeout=1800)
    if out:
        print(out[-2000:])
    if err:
        print("--- stderr ---")
        print(err[-2000:])
    # 校验产物
    out, err = run(c, f"ls -la {remote_proj}/dist/pd_pyinstaller_build/")
    print("[build] 产物:")
    print(out.strip() or err.strip())
    # 打 tar.gz
    out, err = run(c, f"cd {remote_proj}/dist && tar czf pd_pyinstaller_build.tar.gz pd_pyinstaller_build && ls -lh pd_pyinstaller_build.tar.gz")
    print("[build] 打包 tar.gz:")
    print(out.strip() or err.strip())
    c.close()
    return 0


def _latest_run_dir() -> str:
    c = connect()
    out, _ = run(c, "ls -dt /home/LENOVO/Pixel_Diff_*/ 2>/dev/null | head -1")
    c.close()
    d = out.strip().rstrip("/")
    if not d:
        raise RuntimeError("未找到打包目录，请先执行 sync")
    return d


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("sync", "all"):
        remote = do_sync()
    else:
        remote = _latest_run_dir()
    if mode in ("build", "all"):
        do_build(remote)
    print(f"[done] 远程目录: {remote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
