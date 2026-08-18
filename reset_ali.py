"""重置阿里云项目代码：删除代码（保留 .venv 环境与 artifacts 运行数据）+ 全量上传本地代码。"""
from __future__ import annotations

import os
import posixpath

import paramiko

# 部署凭据从环境变量读取：PD_ALIYUN_HOST / PD_ALIYUN_USER(默认 root) / PD_ALIYUN_PWD
HOST = os.environ.get("PD_ALIYUN_HOST", "")
USER = os.environ.get("PD_ALIYUN_USER", "root")
PWD = os.environ.get("PD_ALIYUN_PWD", "")
REMOTE = "/home/Pixel_Diff"
LOCAL = __import__("pathlib").Path("E:/Code/Pixel_Diff_s")

# 删除清单（保留 .venv 与 artifacts）
DELETE_ITEMS = [
    "build", "dist", "build_linux_kylin.spec", "build_linux.spec",
    "configs", "docs", "pyproject.toml", "README.md",
    "scripts", "src", "viewer_a.png", "viewer_b.png",
]

# 上传：子目录（递归）+ 顶层文件
SYNC_SUBDIRS = ["src", "scripts", "configs", "docs", "tests", "test_imgs"]
SYNC_FILES = [
    "build_linux_kylin.spec", "build_linux.spec", "pyproject.toml",
    "README.md", "viewer_a.png", "viewer_b.png",
]
EXCLUDE_PARTS = {"__pycache__", ".egg-info", ".venv", "build", "dist"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def connect() -> paramiko.SSHClient:
    if not HOST or not PWD:
        raise SystemExit(
            "缺少部署凭据：请先设置环境变量 PD_ALIYUN_HOST / PD_ALIYUN_PWD"
        )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, 22, USER, PWD, timeout=20)
    return c


def run(c, cmd):
    _i, o, e = c.exec_command(cmd)
    return o.read().decode().strip(), e.read().decode().strip()


def collect_files() -> list:
    files = []
    for sub in SYNC_SUBDIRS:
        base = LOCAL / sub
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
        files.append(LOCAL / name)
    return files


def main() -> int:
    c = connect()

    print("=== 1. 删除远程代码（保留 .venv / artifacts）===")
    for item in DELETE_ITEMS:
        run(c, f"rm -rf {REMOTE}/{item}")
        print(f"  DEL {item}")
    out, _ = run(c, f"ls -la {REMOTE}/")
    print("--- 删除后剩余 ---")
    print(out)

    print("=== 2. 上传本地代码 ===")
    sftp = c.open_sftp()
    files = collect_files()
    print(f"共 {len(files)} 个文件")
    for p in files:
        rel = p.relative_to(LOCAL).as_posix()
        remote = posixpath.join(REMOTE, rel)
        run(c, f"mkdir -p {posixpath.dirname(remote)}")
        sftp.put(str(p), remote)
    sftp.close()

    print("=== 3. 上传后目录 ===")
    out, _ = run(c, f"ls -la {REMOTE}/")
    print(out)

    c.close()
    print("=== 完成 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
