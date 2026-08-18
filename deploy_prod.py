"""部署到阿里云生产（源码态）：全量同步 src/scripts/configs + 重启 run_api。

用法：
    .venv/Scripts/python.exe deploy_prod.py [sync|restart|all]
"""
from __future__ import annotations

import os
import posixpath
import sys
from pathlib import Path

import paramiko

# 部署凭据从环境变量读取，避免在代码中硬编码生产密码：
#   PD_ALIYUN_HOST  阿里云服务器地址（必填）
#   PD_ALIYUN_USER  登录用户（默认 root）
#   PD_ALIYUN_PWD   登录密码（必填）
HOST = os.environ.get("PD_ALIYUN_HOST", "")
USER = os.environ.get("PD_ALIYUN_USER", "root")
PWD = os.environ.get("PD_ALIYUN_PWD", "")
REMOTE_ROOT = "/home/Pixel_Diff"
LOCAL_ROOT = Path(__file__).resolve().parent

# 需要同步的子目录（全量 .py/.yaml/.txt，排除缓存与打包残留）
SYNC_SUBDIRS = ["src", "scripts", "configs"]
EXCLUDE_PARTS = {"__pycache__", ".venv", "build", "dist"}
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


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 120) -> tuple[str, str]:
    _i, o, e = c.exec_command(cmd, timeout=timeout)
    return o.read().decode(), e.read().decode()


def collect_files() -> list[Path]:
    files: list[Path] = []
    for sub in SYNC_SUBDIRS:
        base = LOCAL_ROOT / sub
        if not base.is_dir():
            continue
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
    return files


def mkdir_p(c: paramiko.SSHClient, remote_dir: str) -> None:
    if remote_dir == "" or remote_dir == "/":
        return
    c.exec_command(f"mkdir -p {remote_dir}")


def do_sync() -> int:
    c = connect()
    sftp = c.open_sftp()
    files = collect_files()
    print(f"[sync] 共 {len(files)} 个文件待同步")
    n_ok = 0
    for p in files:
        rel = p.relative_to(LOCAL_ROOT).as_posix()
        remote = posixpath.join(REMOTE_ROOT, rel)
        remote_dir = posixpath.dirname(remote)
        mkdir_p(c, remote_dir)
        sftp.put(str(p), remote)
        n_ok += 1
        print(f"  PUT {rel}")
    sftp.close()
    print(f"[sync] 完成，成功 {n_ok} 个文件")
    c.close()
    return 0


def do_restart() -> int:
    import time

    c = connect()
    # 一条命令完成 kill + 启动：`grep -v $$` 排除执行 pgrep 的 shell 自身，
    # 避免误杀导致 SSH 通道异常；nohup 加 < /dev/null 避免 exec 卡住。
    restart_cmd = (
        "cd /home/Pixel_Diff && export LD_LIBRARY_PATH=/opt/python3.12/lib && "
        "pgrep -f run_api.py | grep -v $$ | xargs -r kill -9 2>/dev/null; "
        "sleep 2; "
        "nohup .venv/bin/python scripts/run_api.py --host 0.0.0.0 --port 8000 "
        "> /tmp/run_api.log 2>&1 < /dev/null & "
        "echo RESTART_DONE"
    )
    out, err = run(c, restart_cmd, timeout=60)
    print(out or err)
    time.sleep(6)
    # 健康检查
    out, err = run(c, "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/v1/settings || true", timeout=20)
    print(f"[restart] 健康检查 /api/v1/settings -> HTTP {out or err}")
    c.close()
    return 0


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("sync", "all"):
        do_sync()
    if mode in ("restart", "all"):
        do_restart()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
