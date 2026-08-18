# Pixel-Diff 配准与识别流程

## 总览

将一份**待检文档**（扫描件/电子 PDF/图片/DOCX）与一份**审批通过模板**（电子 PDF/图片/DOCX）进行像素级比对，检出所有内容差异（增、删、改、位移），输出差异区域列表、可视化对比图、HTML/DOCX 报告。

核心设计：**以模板坐标为准**，将扫描件通过多级配准逐步对齐到模板坐标系，再做逐像素 XOR 差异检测，最后通过多级过滤管线剔除假阳性。

---

## 阶段 0：CLI 入口预处理 `scripts/compare.py`

```
compare.py <scan_path> <template_path> [--config ...] [--report-dir ...]
```

### 0a. 加载配置
- 从 YAML 加载 `PixelDiffConfig`（默认 `configs/default.yaml`）
- 与 `configs/sensitive_recall_trial.yaml` 一起构成不同召回策略

### 0b. DOCX 自动转换
- DOCX 文件 → `.docx` → `convert_docx_to_pdf()` → PDF，统一后续渲染口径

### 0c. 图片方向归一化 `_normalize_image_orientation()`
- 若 scan 与 template 都是图片且横竖方向不同：
  - 旋转 scan 90° 使其方向与 template 一致
  - 保存到 `_rotated/rotated_<name>`

### 0d. 两种运行模式
| 模式 | 入口 | 并行 |
|---|---|---|
| 单页模式 | `compare()` 直接调用 | 单进程 |
| 报告模式 | ProcessPoolExecutor 逐页并行 | 最多 4 进程 |

---

## 阶段 1：渲染与输入 `engine.py` → `io.py`

```
load_document_page_bgr(path, page, dpi=300) → BGR uint8 图像
```

- PDF：PyMuPDF 渲染 → 300 DPI → BGR
- 图片：cv2.imread → BGR
- 记录模板尺寸，后续所有坐标以此为基准

同时构建两个辅助掩码：
- `build_unchanged_text_mask()` — PDF 文字层中与模板完全相同的文本区域
- `build_pdf_image_keep_mask()` — PDF 图像资源区域（图片/印章）

---

## 阶段 2：预处理 — 颜色过滤与文本锚点对齐 `engine.py`

### 2a. 颜色过滤 `color_filter.py`
```
remove_colored_marks_bgr() → 去除红章蓝签
```

| 目标 | HSV 范围 |
|---|---|
| 红色（公章） | H∈[0,10]∪[170,180], S≥40, V≥40 |
| 蓝色（手写签名） | H∈[100,124], S≥40, V≥40 |

**原理**：扫描件上的红章/蓝签是实物加盖，模板里没有 → XOR 后会产生大片假阳性。过滤后的印章区域填充白色（=空白），不再产生差异。

受保护的不会被过滤：`unchanged_text_mask` 标记了的相同文字区域。

### 2b. 文本锚点粗对齐 `text_anchor_alignment.py`
```
extract_text_anchor_lines() → align_by_text_anchors_bgr()
```

- 从 PDF 文字层提取文本锚点（相同文本行在两份文件中的坐标偏移）
- 若无 PDF 文字层 → `extract_ocr_text_anchor_lines()`（RapidOCR 识别）
- 做平均行间平移校正，为后续 SURF 配准提供更好的初始位姿

---

## 阶段 3：配准 — 五级级联对齐 `alignment.py` → `line_alignment.py` → `local_warp.py`

**这是整个流程的核心**，五级级联将扫描件逐步变换到模板坐标系：

### 3a. SURF/SIFT 全局单应性 `alignment.py`
```
align_scan_to_template_bgr()
```

| 步骤 | 操作 | 参数 |
|---|---|---|
| 1 | SURF 特征提取 | `hessianThreshold=400` |
| 2 | FLANN 匹配 | KD-Tree, k=2 近邻 |
| 3 | Lowe 比率测试 | `d1/d2 < 0.70`（剔除模糊匹配） |
| 4 | RANSAC 单应性估计 | `reprojection_threshold=3.0px` |
| 5 | warpPerspective | INTER_LINEAR + 白边填充 |

**降采样加速**：`alignment_feature_scale=0.5` → 在 1/2 尺寸上提取特征 → 还原到全尺寸
**回退策略**：SURF 不可用 → SIFT；降采样内点率不足 → 回退到全尺寸
**空白页检测**：`blank_page_max_ink_ratio=0.01` → 白页跳过 SURF

校验：`min_good_matches=15`（不够 → AlignmentError）

### 3b. 文本行质心对齐 `line_alignment.py`
```
align_text_lines_by_centroid_bgr()
```

- 提取两幅二值图中每一行文字的质心坐标
- 匹配对应行 → 计算行间平均偏移量
- 逐行做平移校正（补偿扫描仪的 Y 轴非线性偏移）

### 3c. 文本行仿射对齐 `line_affine_alignment.py`
```
align_text_lines_affine_bgr()
```

- 对每行匹配文字做仿射变换校正（平移 + 旋转 + 缩放）
- 校验：before_after IoU 必须提升（IOR≥0.7）

### 3d. 分段局部对齐 `line_piecewise_alignment.py`
```
align_text_lines_piecewise_bgr()
```

- 将页面按文字区域分成若干块（rigid text blocks）
- 每块独立做仿射变换 —— 补偿扫描仪不同区域的局部扭曲

### 3e. 约束局部形变 `local_warp.py`
```
apply_constrained_local_warp_bgr()
```

- 对每个差异候选区域做局部形变匹配
- 校验：形变后的前景 IoU 必须提升 → 否则跳过该区域（`gate_foreground_iou ≥ 0.9`）
- 受 `max_displacement` 约束（默认 20px），防止过度形变

### 3f. 残余行级重对齐 `residual_line_alignment.py`
```
realign_residual_text_lines_bgr()
```

- 在 XOR 差异检测后，用差异掩码区域的对比度引导残余微调
- 保护区间（`protected_intervals`）：锚点附近不调整，防止锚点附近的真实差异被抹除

---

## 阶段 4：二值化 `binarization.py`

扫描件和模板采用**不同策略**：

| 图像 | 策略 | 原因 |
|---|---|---|
| 扫描件 | 双边滤波 → 中值滤波 → 自适应高斯阈值 | 光照不均 + 传感器噪声 + 纸张纹理 |
| 模板 | Otsu 全局阈值 | 纯白背景 + 纯黑文字，无噪声 |

**约定**：前景（文字）= 0（黑），背景（空白）= 255（白）
- 这是有意设计的**反向约定**：XOR 后 0 XOR 255 = 255 → 差异信号

---

## 阶段 5：差异检测 `differ.py` + `morphology.py`

### 5a. 逐像素 XOR 异或
```
xor_difference(scan_binary, template_binary)
```

| 扫描件 | 模板 | XOR | 含义 |
|---|---|---|---|
| 0（字） | 0（字） | 0 | 无差异 |
| 255（空） | 255（空） | 0 | 无差异 |
| 0（字） | 255（空） | **255** | **疑似新增** |
| 255（空） | 0（字） | **255** | **疑似删除** |

### 5b. 边缘裁剪
```
crop_edges(diff, margin=40)  → 四周裁掉 40px
```
抑制扫描仪边框伪影（黑边、装订孔、盖板漏光）

### 5c. 形态学清理 `morphology.py`
```
clean_difference_mask()
```

| 步骤 | 核大小 | 作用 |
|---|---|---|
| 去小连通域 | n/a | 删除面积 < 12px² 的孤立噪点 |
| 开运算 | (3,3)×1 | 断开弱连接、去细毛刺 |
| 闭运算 | (3,3)×1 | 填充区域内小孔洞 |
| 膨胀 | (15,10)×1 | 同一行碎片横向合并（适配中文） |

---

## 阶段 6：区域分析与过滤 `regions.py` + `filter_pipeline.py`

### 6a. 轮廓提取
```
extract_regions(diff_mask, min_diff_area) → [DifferenceRegion]
```
- `cv2.findContours(RETR_EXTERNAL)` → 最外层轮廓
- 过滤面积 < `min_diff_area` 的轮廓
- 计算外接矩形 → 裁剪到图像边界
- 排序：(y ↓, x →, area ↓)

### 6b. 多级过滤管线（可选，`multilevel_filter_enabled`）

**Level 1 — 结构过滤**：

| 过滤器 | 作用 |
|---|---|
| 同行小残片合并 | 同一行 Y 坐标差 < 3px，横向间隙 < 20px → 合并 |
| 彩色残差剔除 | 区域内红/蓝色像素比率 ≥ 30% → 剔除（印章残余） |
| 孤立小残片剔除 | 面积小 + 宽高小 + 离最近区域 > 100px → 剔除 |
| 局部相似性过滤 | 7 级平移搜索，配准残余 → 丢弃 |

**Level 2 — SSIM 结构相似性过滤**：

```
best_ssim_for_region() → 在 padding 范围内搜索平移后 SSIM ≥ 0.95 的位置
```
- 若 SSIM 达标 → 差异来自配准残余，丢弃
- 支持提前退出（`early_exit`）和有缓存模板统计（加速）

**Level 3 — 文本标注**：

```
compare_text_regions() → PDF 文字层文本差异标注
```

### 6c. 文本层差分 `text_layer.py`
```
extract_text_difference_regions() → 从 PDF 文字层直接提取文本增删
merge_regions() → 与像素差异区域合并
filter_recalled_similarity_regions() → 二次过滤假阳性
```

---

## 阶段 7：风险复核 `risk_review.py`

```
apply_risk_review()
```

根据配置文件中的规则（如 `sensitive_recall_trial.yaml` 的 `risk_review_*`），将差异区域分为三个风险等级：

| 风险等级 | 典型场景 | 展示颜色 |
|---|---|---|
| HIGH | 关键金额、日期、签名区域变化 | 红色 |
| MEDIUM | 一般内容变化 | 橙色 |
| LOW | 背景水印变化（`watermark` 配置过滤）、疑似配准残余 | 蓝色 |

### 7a. 位移配对 `displacement.py`
```
pair_displaced_regions() → 识别偏移量相近的多个差异块
```
- 同一段文字整体位移 → 配对为"移位"类型
- 避免把"整段位移"拆成"多处删+加"

### 7b. 差异分类 `change_classification.py`
```
classify_difference_regions() → 自动标记差异类型
```
- `addition` / `deletion` / `modification` / `displacement`
- 分类标签可选覆盖到可视化输出

---

## 阶段 8：可视化输出 `visualization.py` + `report.py`

| 输出 | 格式 | 内容 |
|---|---|---|
| 红框标注图 | PNG | 模板上叠加差异区域彩色边框 |
| 文字残影图 | PNG | 模板文字背景叠加待检文字前景，文字偏色 = 差异 |
| 候选 patch | PNG | 每处差异的局部裁剪图（可选） |
| JSON 结果 | JSON | 差异列表 + 配准指标 + 耗时 |
| HTML 报告 | HTML | 交互式三栏对比视图（模板 \| 残影 \| 差异列表） |
| DOCX 报告 | DOCX | 图文嵌合的可打印报告 |

---

## 完整耗时分解

每次比对的 `metrics` 字段包含各阶段耗时（毫秒）：

| 阶段 | 指标字段 | 耗时级 |
|---|---|---|
| 渲染 | `timing_render_ms` | ~50ms |
| 颜色过滤 | `timing_color_filter_ms` | ~10ms |
| 文本锚点对齐 | `timing_text_anchor_alignment_ms` | ~50ms |
| SURF 全局配准 | `timing_global_alignment_ms` | ~80ms |
| 文本行对齐 | `timing_line_alignment_ms` | ~20ms |
| 分段对齐 | `timing_piecewise_alignment_ms` | ~30ms |
| 局部形变 | `timing_local_warp_ms` | ~3ms |
| 二值化 | `timing_binarization_ms` | ~30ms |
| 差异检测 | `timing_difference_text_ms` | ~50ms |
| 过滤管线 | `timing_filtering_ms` | ~40ms |
| 风险复核 | `timing_risk_review_ms` | ~10ms |
| 可视化输出 | `timing_output_ms` | ~5ms |
| **总计** | `elapsed_ms` | **~400ms**（单页 300DPI） |

---

## 关键配置参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `dpi` | 300 | 渲染精度 |
| `feature_detector` | surf | 特征检测器（surf/sift） |
| `lowe_ratio` | 0.70 | Lowe 比率测试阈值 |
| `min_good_matches` | 15 | 最少匹配点数 |
| `ransac_reprojection_threshold` | 3.0 | RANSAC 重投影误差 |
| `crop_margin` | 40 | 边缘裁剪像素 |
| `min_diff_area` | 12 | 最小差异面积（px²） |
| `ssim_filter_threshold` | 0.95 | SSIM 过滤阈值 |
| `multilevel_filter_enabled` | True | 启用三级过滤管线 |
| `risk_review_enabled` | True | 启用风险复核 |
| `line_centroid_alignment` | True | 启用文本行质心对齐 |
| `local_warp_enabled` | True | 启用约束局部形变 |

---

## 异常处理与快速失败

| 条件 | 异常/状态 | HTTP 状态 | 说明 |
|---|---|---|---|
| 特征点不足 | `AlignmentError` | 50002 | good_matches < 15 |
| 单应性失败 | `AlignmentError` | 50002 | RANSAC 无法收敛 |
| 页数不匹配 | `InputError` | - | scan 与 template 页数不同 |
| **差异率 > 50%** | `task.failed` | 50002 | 两份文档几乎完全不同 → shortcut 为失败，展示失败对照页 |
| 描述子为空 | `AlignmentError` | 50002 | 图像无结构特征 |
