"""验证阿里云打包产物：含最新修复、启动可用、打 tar.gz，并精确清理测试进程。"""
from __future__ import annotations

import os
import time

import paramiko

# 部署凭据从环境变量读取：PD_ALIYUN_HOST / PD_ALIYUN_USER(默认 root) / PD_ALIYUN_PWD
HOST = os.environ.get("PD_ALIYUN_HOST", "")
USER = os.environ.get("PD_ALIYUN_USER", "root")
PASS = os.environ.get("PD_ALIYUN_PWD", "")
PORT = 8005  # 产物测试端口，避开 8000 生产


def main() -> int:
    if not HOST or not PASS:
        raise SystemExit(
            "缺少部署凭据：请先设置环境变量 PD_ALIYUN_HOST / PD_ALIYUN_PWD"
        )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, 22, USER, PASS, timeout=20)

    def run(cmd: str, t: int = 30) -> str:
        _i, o, e = c.exec_command(cmd, timeout=t)
        return (o.read().decode() + e.read().decode()).strip()

    # 1. 产物内 task_service.py 是否含最新修复（_input_name / template_path 兜底）
    out = run(
        'grep -c "_input_name" /home/Pixel_Diff/dist/pd_pyinstaller_build/_internal/'
        "pixel_diff_api/task_service.py 2>/dev/null"
    )
    print(f"[1] 产物含 _restore 修复标记(_input_name): {out or '未找到'}")

    # 2. 打 tar.gz
    out = run(
        "cd /home/Pixel_Diff/dist && rm -f pd_pyinstaller_build.tar.gz && "
        "tar czf pd_pyinstaller_build.tar.gz pd_pyinstaller_build && "
        "ls -lh pd_pyinstaller_build.tar.gz"
    )
    print(f"[2] tar.gz:\n{out}")

    # 3. 用产物启动测试（8005）
    run(
        f"cd /home/Pixel_Diff/dist/pd_pyinstaller_build && "
        "export LD_LIBRARY_PATH=/opt/python3.12/lib && "
        f"nohup ./run_api --host 127.0.0.1 --port {PORT} > /tmp/run_api_{PORT}.log 2>&1 < /dev/null &"
    )
    time.sleep(6)
    out = run(f"curl -s http://127.0.0.1:{PORT}/api/v1/settings || echo CURL_FAIL")
    print(f"[3] 产物 /api/v1/settings: {out[:200]}")
    out = run(f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{PORT}/api/v1/system/memory")
    print(f"[3] 产物 /api/v1/system/memory: HTTP {out}")

    # 4. 精确清理产物测试进程（避免 pgrep -f 匹配到自身 shell）
    out = run(f"pgrep -f 'port {PORT}' | xargs -r kill -9 2>/dev/null; sleep 1; "
              f"pgrep -f 'port {PORT}' || echo CLEANED")
    print(f"[4] 测试进程清理: {out}")

    # 5. 确认生产服务仍健康
    out = run("curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/api/v1/settings")
    print(f"[5] 生产 8000 服务: HTTP {out}")
    c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
