# KylinV10 ARM64 打包指南

> 适用系统：麒麟 V10 / V11（ARM64 aarch64）

> 目标 CPU：飞腾 FT-2000 / S2500 / 鲲鹏 920

> 打包方式：PyInstaller 、 Nuitka

---

## 目录

1.  [环境准备（ARM64 专用）](#1-%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87arm64-%E4%B8%93%E7%94%A8)
    
2.  [方式一：PyInstaller 打包（推荐）](#2-%E6%96%B9%E5%BC%8F%E4%B8%80pyinstaller-%E6%89%93%E5%8C%85%E6%8E%A8%E8%8D%90)
    
3.  [方式二：Nuitka 打包（体积更小）](#3-%E6%96%B9%E5%BC%8F%E4%BA%8Cnuitka-%E6%89%93%E5%8C%85%E4%BD%93%E7%A7%AF%E6%9B%B4%E5%B0%8F)
    
4.  [产物验证与运行](#4-%E4%BA%A7%E7%89%A9%E9%AA%8C%E8%AF%81%E4%B8%8E%E8%BF%90%E8%A1%8C)
    
5.  [打包导出](#5-%E6%89%93%E5%8C%85%E5%AF%BC%E5%87%BA)
    
6.  [ARM64 常见问题](#6-arm64-%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98)
    

---

## 1. 环境准备（ARM64 专用）

### 1.1 确认架构

```plaintext
uname -m
# 输出：aarch64  ✅（ARM64 架构）
# 如果输出：x86_64  → 走 x86_64 版指南

```

### 1.2 安装系统包

```plaintext
# --- ARM64 基础工具链 ---
sudo dnf install -y \
    gcc gcc-c++ make git wget curl \
    python3 python3-devel python3-pip python3-venv \
    patchelf mesa-libGL libGLU libffi-devel \
    openssl-devel bzip2-devel zlib-devel xz-devel sqlite-devel

# --- LibreOffice（DOCX→PDF 转换，建议安装）---
sudo dnf install -y libreoffice || echo "跳过 LibreOffice"

```

### 1.3 检查 Python 版本

```plaintext
python3 --version
# 需要 ≥3.11，如果低于 3.11，用 uv 升级

```

### 1.4 安装 uv 与 Python 3.12（可选，用于升级 Python）

```plaintext
# 麒麟 ARM64 预装 Python 3.9 的需升级
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
# 装 ARM64 Python 3.12
uv python install 3.12
uv venv --python 3.12 .venv

# 如果 uv 的 ARM64 版本有问题，手动编译 Python 3.12（见 6.6 节）

```

### 1.5 配置 pip 镜像（ARM64 网络较慢，必须配）

```plaintext
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << 'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF

```

### 1.6 安装项目依赖

```plaintext
cd /root/Pixel_Diff
source .venv/bin/activate

# 分批安装避免超时
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ numpy
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ opencv-contrib-python
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ PyMuPDF pypdfium2 PyYAML
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ fastapi 'uvicorn[standard]' python-multipart
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ -e .

# --- 如果 opencv-contrib-python 没有 ARM64 wheel ---
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ opencv-python
# 然后在 pyproject.toml 中把 opencv-contrib-python 改为 opencv-python

```
---

## 2. 方式一：PyInstaller 打包

**优点**：稳定、构建快、无需编译 C 代码**缺点**：体积较大（~200 MB）

### 2.1 安装 PyInstaller

```plaintext
source .venv/bin/activate
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ pyinstaller

```

### 2.2 创建 build\_linux.spec

```plaintext
cat > /root/Pixel_Diff/build_linux.spec << 'SPECEOF'
# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller ARM64 Linux 打包脚本"""
from pathlib import Path
ROOT = Path(SPECPATH).resolve()
SRC = ROOT / "src"
CONFIGS = ROOT / "configs"

HIDDEN = [
    "pixel_diff","pixel_diff._app_paths","pixel_diff.engine","pixel_diff.models",
    "pixel_diff.exceptions","pixel_diff.io","pixel_diff.alignment",
    "pixel_diff.text_anchor_alignment","pixel_diff.line_alignment",
    "pixel_diff.line_affine_alignment","pixel_diff.line_piecewise_alignment",
    "pixel_diff.rigid_text_block_alignment","pixel_diff.residual_line_alignment",
    "pixel_diff.local_warp","pixel_diff.color_filter","pixel_diff.text_layer",
    "pixel_diff.binarization","pixel_diff.differ","pixel_diff.morphology",
    "pixel_diff.regions","pixel_diff.filter_pipeline","pixel_diff.similarity",
    "pixel_diff.risk_review","pixel_diff.change_classification",
    "pixel_diff.displacement","pixel_diff.patch_export",
    "pixel_diff.visualization","pixel_diff.report","pixel_diff.timing",
    "pixel_diff.region_utils",
    "pixel_diff_api","pixel_diff_api.app","pixel_diff_api.settings",
    "pixel_diff_api.task_service","pixel_diff_api.viewer",
]
EXCLUDES = ["rapidocr_onnxruntime","onnxruntime","easyocr","pytesseract",
    "matplotlib","scipy","tkinter","torch","tensorflow","pywin32","win32com","pythoncom"]

a_cli = Analysis([str(ROOT/"scripts"/"compare.py")], pathex=[str(SRC)],

    binaries=[], datas=[(str(CONFIGS),"configs")], hiddenimports=HIDDEN,


    hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=EXCLUDES, noarchive=False)

a_api = Analysis([str(ROOT/"scripts"/"run_api.py")], pathex=[str(SRC)],

    binaries=[], datas=[(str(CONFIGS),"configs")], hiddenimports=HIDDEN,


    hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=EXCLUDES, noarchive=False)

pyz = PYZ(a_cli.pure+a_api.pure, a_cli.zipped_data+a_api.zipped_data)

exe_cli = EXE(pyz, a_cli.scripts, [], exclude_binaries=True, name="compare",

    debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True)

exe_api = EXE(pyz, a_api.scripts, [], exclude_binaries=True, name="run_api",

    debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True)
coll = COLLECT(exe_cli, exe_api, a_cli.binaries, a_api.binaries, a_cli.datas, a_api.datas,

    strip=False, upx=True, upx_exclude=[], name="pd_pyinstaller_build")

import shutil; shutil.copytree(CONFIGS, ROOT/"dist"/"pd_pyinstaller_build"/"configs", dirs_exist_ok=True)
SPECEOF

```
> **与 x86\_64 版本完全一致**——PyInstaller 的 spec 文件跨架构通用，无需修改。

### 2.3 执行打包

```plaintext
cd /root/Pixel_Diff
source .venv/bin/activate

# ARM64 编译设置（发挥飞腾 CPU 性能）
export CFLAGS="-march=armv8-a+crc+crypto -O3"
export CXXFLAGS="-march=armv8-a+crc+crypto -O3"

pyinstaller --clean -y build_linux.spec 2>&1 | tail -30

```

#### 预计时长：ARM64 上约 10-15 分钟

#### 成功标志

```plaintext
INFO: Building COLLECT COLLECT-00.toc completed successfully.

```

### 2.4 产物路径

```plaintext
dist/pd_pyinstaller_build/
├── compare          # CLI 对比工具（ARM64 ELF）
├── run_api          # FastAPI 服务（ARM64 ELF）
├── configs/         # 配置文件（3 个 YAML）
└── _internal/       # ARM64 运行时库（~200 MB）

```
---

## 3. 方式二：Nuitka 打包（体积更小）

**优点**：产物体积小（~30-40 MB）、原生机器码启动更快**缺点**：构建慢（ARM64 上 1-2 小时）、依赖 gcc 版本

### 3.1 安装 Nuitka

```plaintext
source .venv/bin/activate
pip install --index-url https://mirrors.aliyun.com/pypi/simple/ nuitka

# 验证
python -m nuitka --version
# 需要 ≥ 2.x

```

### 3.2 设置飞腾 CPU 编译优化

```plaintext
# 飞腾 FT-2000 / S2500 编译参数（ARM64 必须加）
export PYTHONPATH=/root/Pixel_Diff/src
export CFLAGS="-march=armv8-a+crc+crypto -O3"
export CXXFLAGS="-march=armv8-a+crc+crypto -O3"

```

### 3.3 执行编译

#### 编译 run\_api.bin（API 服务）

```plaintext
python -m nuitka \
    --standalone \
    --lto=yes \
    --remove-output \
    --jobs=8 \
    --include-package=pixel_diff \
    --include-package=pixel_diff_api \
    --output-filename=run_api.bin \
    --nofollow-import-to=rapidocr_onnxruntime \
    --nofollow-import-to=onnxruntime \
    --nofollow-import-to=easyocr \
    --nofollow-import-to=pytesseract \
    --nofollow-import-to=scipy \
    --nofollow-import-to=matplotlib \
    --nofollow-import-to=tkinter \
    --nofollow-import-to=torch \
    --nofollow-import-to=tensorflow \
    --nofollow-import-to=win32com \
    scripts/run_api.py 2>&1 | tail -30

```

#### 编译 compare.bin（CLI 工具）

```plaintext
python -m nuitka \
    --standalone \
    --lto=yes \
    --remove-output \
    --jobs=8 \
    --include-package=pixel_diff \
    --output-filename=compare.bin \
    --nofollow-import-to=rapidocr_onnxruntime \
    --nofollow-import-to=onnxruntime \
    --nofollow-import-to=easyocr \
    --nofollow-import-to=pytesseract \
    --nofollow-import-to=scipy \
    --nofollow-import-to=matplotlib \
    --nofollow-import-to=tkinter \
    --nofollow-import-to=torch \
    --nofollow-import-to=tensorflow \
    --nofollow-import-to=win32com \
    scripts/compare.py 2>&1 | tail -30

```

#### 预计时长：ARM64 上约 1-2 小时

### 3.4 拷贝配置文件

```plaintext
cp -r configs run_api.dist/
cp -r configs compare.dist/

```

### 3.5 产物路径

```plaintext
run_api.dist/
├── run_api.bin          # API 服务（~15 MB）
├── *.so                 # ARM64 动态库
└── configs/

compare.dist/
├── compare.bin          # CLI 工具
├── *.so
└── configs/

```
---

## 4. 产物验证与运行

### 4.1 验证可执行文件架构

```plaintext
file /opt/pixel_diff/run_api
# 输出：... ELF 64-bit LSB executable, ARM aarch64 ...

```

### 4.2 运行 API 服务

```plaintext
# 复制到 /opt（/root 可能有 noexec 限制）
cp -r /root/Pixel_Diff/dist/pd_pyinstaller_build /opt/pixel_diff
chmod -R +x /opt/pixel_diff/

# 启动
/opt/pixel_diff/run_api --host 0.0.0.0 --port 8000

# 后台运行
nohup /opt/pixel_diff/run_api --host 0.0.0.0 --port 8000 \
    > /var/log/pixel-diff-api.log 2>&1 &

```

### 4.3 运行 CLI 对比

```plaintext
/opt/pixel_diff/compare doc_a.pdf doc_b.pdf --report-dir /tmp/result

```

### 4.4 浏览器访问

```plaintext
http://<ARM64_服务器_IP>:8000/docs
http://<ARM64_服务器_IP>:8000/api/v1/compare/tasks

```
---

## 5. 打包导出

### 5.1 麒麟端压缩

```plaintext
# PyInstaller 产物
cd /root/Pixel_Diff/dist
tar czf /tmp/pixel_diff_arm64.tar.gz pd_pyinstaller_build/

# Nuitka 产物
cd /root/Pixel_Diff
tar czf /tmp/pixel_diff_arm64_nuitka.tar.gz run_api.dist/

ls -lh /tmp/pixel_diff_arm64*.tar.gz

```

### 5.2 传输到目标机器

```plaintext
# scp
scp /tmp/pixel_diff_arm64.tar.gz root@目标ARM64机器:/opt/

# 或在目标机解压
ssh root@目标ARM64机器
cd /opt
tar xzf /opt/pixel_diff_arm64.tar.gz
chmod -R +x /opt/pd_pyinstaller_build

```
---

## 6. ARM64 常见问题

### 6.1 opencv-contrib-python 没有 ARM64 wheel

**症状**：`pip install opencv-contrib-python` 在 ARM64 上找不到匹配 wheel

**解决**：

```plaintext
# 改用 opencv-python（标准版，ARM64 有预编译 wheel）
pip install opencv-python

# 修改 pyproject.toml
sed -i 's/opencv-contrib-python/opencv-python/' /root/Pixel_Diff/pyproject.toml

```

### 6.2 Nuitka 编译 ARM64 时 gcc 内存不足（OOM）

**症状**：编译过程中进程被 kill**解决**：

```plaintext
# 减少并行编译数（从 --jobs=8 改为 --jobs=4）
export CFLAGS="-march=armv8-a+crc+crypto -O2"    # 用 -O2 代替 -O3，省内存
python -m nuitka ... --jobs=4 ...

```

### 6.3 `patchelf` 在 ARM64 上不支持

**症状**：`dnf install patchelf` 后 Nuitka 仍报错**解决**：

```plaintext
# 确认装的是 ARM64 版本
rpm -q --queryformat '%{ARCH}\n' patchelf
# 应输出 aarch64

# 从源码编译
git clone https://github.com/NixOS/patchelf.git
cd patchelf && ./bootstrap.sh && ./configure && make && make install

```

### 6.4 Nuitka 编译 run\_api 耗时太长

| CPU | 预计时间 |
| --- | --- |
| 飞腾 FT-2000/4（4 核） | 3-4 小时 |
| 飞腾 S2500（8 核） | 1-1.5 小时 |
| 鲲鹏 920（16 核） | 30-60 分钟 |

→ **推荐使用 PyInstaller**，ARM64 上 10-15 分钟即可完成。

---
---