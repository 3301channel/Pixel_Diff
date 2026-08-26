"""探测 192.168.4.227 目标机环境，确定打包方案。

用法（凭据走环境变量，避免明文写进文件）：
    PD_KYLIN_HOST=192.168.4.227 PD_KYLIN_USER=LENOVO PD_KYLIN_PWD=xxx \
        python probe_227.py
"""
from __future__ import annotations

import os
import sys

import paramiko

HOST = os.environ.get("PD_KYLIN_HOST", "192.168.4.227")
USER = os.environ.get("PD_KYLIN_USER", "LENOVO")
PWD = os.environ.get("PD_KYLIN_PWD", "")
PORT = int(os.environ.get("PD_KYLIN_PORT", "22"))

CMDS = [
    ("架构 uname -m", "uname -m"),
    ("内核 uname -r", "uname -r"),
    ("系统版本", "cat /etc/os-release 2>/dev/null | head -6"),
    ("麒麟版本", "cat /etc/kylin-release 2>/dev/null; cat /etc/.kyinfo 2>/dev/null | head -3"),
    ("CPU 核数", "nproc"),
    ("内存", "free -h | head -2"),
    ("磁盘", "df -h /home 2>/dev/null | tail -1"),
    ("python3", "python3 --version 2>&1; which -a python3 python 2>/dev/null"),
    ("/opt 下 Python", "ls -d /opt/python* 2>/dev/null; ls /opt 2>/dev/null | head -30"),
    ("pip3", "pip3 --version 2>&1; python3 -m pip --version 2>&1"),
    ("pyinstaller", "which pyinstaller 2>/dev/null; python3 -m PyInstaller --version 2>&1 | head -1"),
    ("/home/LENOVO 现有目录", "ls -la /home/LENOVO/ 2>/dev/null"),
    ("已装关键库", "python3 -c 'import numpy, cv2, fitz; print(\"numpy\", numpy.__version__, \"opencv\", cv2.__version__, \"pymupdf\", fitz.__doc__)' 2>&1 | head -3"),
    ("LD 库", "ls /opt/python3.12/lib 2>/dev/null | head; ls /usr/lib64/libpython3* 2>/dev/null"),
]


def main() -> int:
    if not PWD:
        print("缺少 PD_KYLIN_PWD，请先 export 密码", file=sys.stderr)
        return 2
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(HOST, PORT, USER, PWD, timeout=20)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] 连接失败: {e}", file=sys.stderr)
        return 1
    print(f"=== 已连接 {HOST} ===")
    for label, cmd in CMDS:
        _i, o, e = c.exec_command(cmd, timeout=30)
        out = o.read().decode(errors="replace").strip()
        err = e.read().decode(errors="replace").strip()
        print(f"\n--- {label} ---")
        if out:
            print(out)
        if err:
            print(f"[stderr] {err}")
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
