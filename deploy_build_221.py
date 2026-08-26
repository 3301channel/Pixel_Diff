"""部署到 192.168.2.21（x86_64 麒麟服务器，root + Miniconda）并 PyInstaller 打包。

与 deploy_build.py 的区别：
- 目标机是 root 用户、x86_64、麒麟高级服务器版
- Python 环境用 Miniconda（/opt/miniconda3/envs/pd），无需 LD_LIBRARY_PATH
- 代码放独立新目录 /root/Pixel_Diff_build/<时间戳>/

用法（凭据走环境变量）：
    PD_BUILD_HOST=192.168.2.21 PD_BUILD_USER=root PD_BUILD_PWD='1qaz@WSX3edc' \
        python deploy_build_221.py [sync|build|all]
"""
from __future__ import annotations

import datetime
import os
import posixpath
import sys
from pathlib import Path

import paramiko

HOST = os.environ.get("PD_BUILD_HOST", "192.168.2.21")
USER = os.environ.get("PD_BUILD_USER", "root")
PWD = os.environ.get("PD_BUILD_PWD", "")
REMOTE_BASE = os.environ.get("PD_BUILD_BASE", "/root/Pixel_Diff_build")
PYINSTALLER = "/opt/miniconda3/envs/pd/bin/pyinstaller"
RUN_DIR_NAME = "Pixel_Diff_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
REMOTE_PROJ = posixpath.join(REMOTE_BASE, RUN_DIR_NAME)

LOCAL_ROOT = Path(__file__).resolve().parent
SYNC_SUBDIRS = ["src", "scripts", "configs"]
SYNC_FILES = ["build_linux_kylin.spec", "pyproject.toml"]
EXCLUDE_PARTS = {"__pycache__", ".venv", "build", "dist", ".egg-info"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def connect() -> paramiko.SSHClient:
    if not PWD:
        raise SystemExit("缺少 PD_BUILD_PWD，请先设置环境变量")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, 22, USER, PWD, timeout=20)
    return c


def run(c: paramiko.SSHClient, cmd: str, timeout: int | None = None):
    _i, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(errors="replace"), e.read().decode(errors="replace")


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
        p = LOCAL_ROOT / name
        if p.is_file():
            files.append(p)
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
        run(c, f"mkdir -p {posixpath.dirname(remote)}")
        sftp.put(str(p), remote)
    sftp.close()
    c.close()
    return REMOTE_PROJ


def do_build(remote_proj: str | None = None) -> int:
    remote_proj = remote_proj or _latest_run_dir()
    c = connect()
    build_cmd = (
        f"cd {remote_proj} && {PYINSTALLER} --clean -y build_linux_kylin.spec"
    )
    print(f"[build] 开始打包（x86_64 2 核，预计 3~6 分钟）...")
    print(f"[build] 命令: {build_cmd}")
    out, err = run(c, build_cmd, timeout=1800)
    if out:
        print(out[-2500:])
    if err:
        print("--- stderr ---")
        print(err[-2500:])
    out, err = run(c, f"ls -la {remote_proj}/dist/pd_pyinstaller_build/")
    print("[build] 产物:")
    print(out.strip() or err.strip())
    out, err = run(
        c,
        f"cd {remote_proj}/dist && tar czf pd_pyinstaller_build.tar.gz "
        f"pd_pyinstaller_build && ls -lh pd_pyinstaller_build.tar.gz",
    )
    print("[build] tar.gz:")
    print(out.strip() or err.strip())
    c.close()
    return 0


def _latest_run_dir() -> str:
    c = connect()
    out, _ = run(c, f"ls -dt {REMOTE_BASE}/Pixel_Diff_*/ 2>/dev/null | head -1")
    c.close()
    d = out.strip().rstrip("/")
    if not d:
        raise RuntimeError("未找到打包目录，请先执行 sync")
    return d


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    remote = None
    if mode in ("sync", "all"):
        remote = do_sync()
    if mode in ("build", "all"):
        do_build(remote)
    print(f"[done] 远程目录: {remote or _latest_run_dir()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
