"""上传代码到阿里云服务器 /home/Pixel_Diff_s/ 并打包（PyInstaller）。

步骤：
  1) 同步本项目顶层必需文件到 /home/Pixel_Diff_s/（src/scripts/configs + spec + pyproject）
  2) 复制已装好的 venv 到 /home/Pixel_Diff_s/.venv 复用（含 PyInstaller + 依赖）
  3) 在 /home/Pixel_Diff_s/ 下执行 build_linux_kylin.spec 打包

用法：
    .venv/Scripts/python.exe deploy_build_server.py [sync|venv|build|all]
"""
from __future__ import annotations

import os
import posixpath
import sys
import time
from pathlib import Path

import paramiko

# 部署凭据从环境变量读取：PD_ALIYUN_HOST / PD_ALIYUN_USER(默认 root) / PD_ALIYUN_PWD
HOST = os.environ.get("PD_ALIYUN_HOST", "")
USER = os.environ.get("PD_ALIYUN_USER", "root")
PWD = os.environ.get("PD_ALIYUN_PWD", "")
REMOTE_ROOT = "/home/Pixel_Diff_s"
SRC_VENV = "/home/Pixel_Diff/.venv"
LOCAL_ROOT = Path(__file__).resolve().parent

# 需要同步的目录与顶层文件（其余本地工具/大目录不传）
SYNC_DIRS = ["src", "scripts", "configs"]
SYNC_FILES = ["build_linux_kylin.spec", "pyproject.toml", "README.md"]
EXCLUDE_PARTS = {"__pycache__", ".venv", "build", "dist", "artifacts", "outputs", "test_imgs", "tests", "docs", ".workbuddy"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def connect() -> paramiko.SSHClient:
    if not HOST or not PWD:
        raise SystemExit(
            "缺少部署凭据：请先设置环境变量 PD_ALIYUN_HOST / PD_ALIYUN_PWD"
            "（USER 默认 root，可用 PD_ALIYUN_USER 覆盖）"
        )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, 22, USER, PWD, timeout=20)
    return c


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 600) -> tuple[str, str]:
    i, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(), e.read().decode()


def collect_files() -> list[Path]:
    files: list[Path] = []
    for d in SYNC_DIRS:
        base = LOCAL_ROOT / d
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_PARTS for part in p.parts):
                continue
            if p.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            files.append(p)
    for f in SYNC_FILES:
        p = LOCAL_ROOT / f
        if p.is_file():
            files.append(p)
    return files


def do_sync() -> int:
    c = connect()
    sftp = c.open_sftp()
    files = collect_files()
    print(f"[sync] {len(files)} 个文件待上传到 {REMOTE_ROOT}")
    run(c, f"mkdir -p {REMOTE_ROOT}")
    n_ok = 0
    for p in files:
        rel = p.relative_to(LOCAL_ROOT).as_posix()
        remote = posixpath.join(REMOTE_ROOT, rel)
        remote_dir = posixpath.dirname(remote)
        c.exec_command(f"mkdir -p {remote_dir}")
        sftp.put(str(p), remote)
        n_ok += 1
    sftp.close()
    print(f"[sync] 完成，成功 {n_ok} 个文件")
    c.close()
    return 0


def do_venv() -> int:
    c = connect()
    print(f"[venv] 复制 {SRC_VENV} -> {REMOTE_ROOT}/.venv")
    out, err = run(c, f"rm -rf {REMOTE_ROOT}/.venv && cp -a {SRC_VENV} {REMOTE_ROOT}/.venv && echo VENV_OK")
    print(out.strip(), err.strip())
    # 校验
    out, err = run(c, f"export LD_LIBRARY_PATH=/opt/python3.12/lib; {REMOTE_ROOT}/.venv/bin/pyinstaller --version")
    print("[venv] pyinstaller version:", out.strip() or err.strip())
    c.close()
    return 0


def do_build() -> int:
    c = connect()
    build_cmd = (
        f"cd {REMOTE_ROOT} && "
        "export LD_LIBRARY_PATH=/opt/python3.12/lib && "
        "rm -rf build dist && "
        ".venv/bin/pyinstaller --clean -y build_linux_kylin.spec"
    )
    print("[build] 开始打包（约 40s）...")
    out, err = run(c, build_cmd, timeout=600)
    print(out)
    if err.strip():
        print("STDERR:", err.strip())
    # 校验产物
    out, err = run(c, f"ls -la {REMOTE_ROOT}/dist/pd_pyinstaller_build/ && echo BUILD_DONE")
    print(out.strip())
    c.close()
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("sync", "all"):
        do_sync()
    if mode in ("venv", "all"):
        do_venv()
    if mode in ("build", "all"):
        do_build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
