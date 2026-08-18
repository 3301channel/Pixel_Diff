"""下载阿里云打包产物 tar.gz 到本地桌面，并清理测试进程。"""
from __future__ import annotations

import os

import paramiko

# 部署凭据从环境变量读取：PD_ALIYUN_HOST / PD_ALIYUN_USER(默认 root) / PD_ALIYUN_PWD
HOST = os.environ.get("PD_ALIYUN_HOST", "")
USER = os.environ.get("PD_ALIYUN_USER", "root")
PWD = os.environ.get("PD_ALIYUN_PWD", "")
REMOTE_TAR = "/home/Pixel_Diff/dist/pd_pyinstaller_build.tar.gz"
LOCAL_TAR = "C:/Users/Jason/Desktop/pd_pyinstaller_build.tar.gz"


def main() -> int:
    if not HOST or not PWD:
        raise SystemExit(
            "缺少部署凭据：请先设置环境变量 PD_ALIYUN_HOST / PD_ALIYUN_PWD"
        )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, 22, USER, PWD, timeout=20)

    # 清理 8003 测试进程（避免残留）
    i, o, e = c.exec_command("pgrep -f 'run_api --host 127.0.0.1 --port 8003' | xargs -r kill -9 2>/dev/null; echo cleaned")
    o.read()

    print(f"[download] {REMOTE_TAR} -> {LOCAL_TAR}")
    sftp = c.open_sftp()
    sftp.get(REMOTE_TAR, LOCAL_TAR)
    sftp.close()
    c.close()
    print("[download] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
