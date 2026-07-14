# Pixel-Diff 改动过程与算法说明

本文记录本轮 Pixel-Diff 项目的主要改动过程，并说明当前流程中涉及的图像处理方法、关键参数和误判过滤策略。本文面向后续维护和人工复核场景，重点说明系统为什么这样处理，而不是给出法律意义上的篡改结论。

## 1. 项目定位

Pixel-Diff 的目标是比对两类文件：

- 模板文件：审批通过的电子 PDF、标准化图片或可转换为 PDF 的 DOCX。
- 待检文件：打印、签署、扫描或被修改后的 PDF、图片或 DOCX。

系统输出的是疑似差异区域坐标和可视化标注图，供人工复核使用。当前实现不做合同效力判断，不做司法或合规结论，也不把检测结果等同于篡改结论。

## 2. 本轮改动概览

本轮改动主要围绕可运行性、报告展示、多页处理和误判控制展开：

1. 支持每次运行自动创建独立结果目录，避免覆盖历史结果。
2. 支持 DOCX 输入转换为 PDF 后参与比对。
3. PDF 渲染优先使用 `pypdfium2`，在 PyMuPDF DLL 受限时仍可运行。
4. 报告输出改为中文，并支持多页 PDF 全页检测。
5. HTML 报告按页展示原始文档、待检文档、比对结果和坐标表。
6. 文字残影图中增加 box 框和序号，便于人工核对。
7. 引入自适应二值化、双边滤波、中值滤波、连通域过滤和形态学处理，降低扫描噪声。
8. 基于 `document1`、`document2` 样本持续调试误判，增加局部相似度、横线残影、稀疏小噪点和宽文本行残影过滤。

## 3. 当前处理流程

当前单页处理流程如下：

```text
输入文件
  -> PDF/DOCX/图片读取
  -> 页面渲染为 OpenCV BGR 图像
  -> HSV 过滤红章、蓝色签字
  -> SIFT 特征提取
  -> FLANN 特征匹配
  -> Lowe 比率测试
  -> RANSAC 估计单应性矩阵
  -> 将待检图配准到模板尺寸
  -> 灰度化、滤波、自适应二值化
  -> XOR 差分
  -> 边缘裁剪
  -> 连通域小噪点去除
  -> 形态学开、闭、膨胀处理
  -> 连通域提取和排序
  -> 残影误判过滤
  -> JSON 坐标、标注图、HTML 报告
```

多页报告模式下，系统会对两份文档的每一页重复上述流程。页数不一致时会报错，不把缺页当作“无差异”。

## 4. PDF 和 DOCX 输入处理

### 4.1 PDF 渲染

系统将 PDF 页面渲染为图像后再进行像素级比对。默认 DPI 为 300，可以通过配置调整。

本轮改动后，PDF 渲染优先使用 `pypdfium2`。原因是部分 Windows 环境会阻止 PyMuPDF 原生 DLL 加载，导致报错：

```text
PyMuPDF is unavailable; PDF rendering requires a working PyMuPDF installation and permission to load its native DLL
```

引入 `pypdfium2` 后，即使 PyMuPDF 不可用，PDF 仍可正常渲染。

### 4.2 DOCX 转换

DOCX 本身不是当前图像比对流程的直接输入格式。现在的处理方式是：

1. 若输入为 `.docx`，先转换为临时 PDF。
2. 转换后的 PDF 再进入标准渲染和比对流程。
3. 转换产物写入本次运行目录下的 `_converted` 子目录。

Windows 环境下优先使用可用的 LibreOffice 或 Microsoft Word COM 能力进行转换。

## 5. 颜色过滤

扫描件中常见的红色印章和蓝色签字会造成大量非正文差异。系统在配准前先对待检图执行 HSV 颜色过滤。

默认过滤对象：

- 红色印章：配置两个 HSV 区间，覆盖色相 0 附近和 180 附近的红色。
- 蓝色签字：配置蓝色 HSV 区间。

过滤方式不是删除整块区域，而是把符合颜色范围的像素替换为背景色，降低其对 SIFT 配准和后续差分的干扰。

注意：黑色签字无法仅凭 HSV 稳定识别，当前不把黑色签名过滤作为默认能力。

## 6. 特征点配准

### 6.1 为什么需要配准

扫描件和电子模板即使内容一致，也可能存在：

- 页面缩放差异。
- 旋转差异。
- 平移偏差。
- 扫描时产生的轻微透视变形。

如果不先配准，后续 XOR 会把整页文字边缘都识别为差异。

### 6.2 SIFT

当前 MVP 默认使用 SIFT 进行关键点检测和描述子提取。SIFT 的优点是对尺度变化、旋转变化和一定光照变化较稳定，适合处理扫描件和模板之间的全局几何偏差。

流程如下：

1. 将模板图和过滤后的待检图转为灰度。
2. 使用 SIFT 提取关键点和描述子。
3. 若任一侧描述子为空，直接报配准失败。
4. 若有效匹配点不足，直接报配准失败。

对应实现位置：

- `src/pixel_diff/alignment.py`

### 6.3 FLANN 和 Lowe 比率测试

SIFT 描述子是高维浮点向量。系统使用 FLANN 做近似最近邻匹配，提高匹配效率。

每个待检描述子会找两个最近邻：

- 最近邻距离记为 `d1`
- 次近邻距离记为 `d2`

若满足：

```text
d1 < lowe_ratio * d2
```

则认为该匹配相对可靠。默认 `lowe_ratio` 为 `0.70`。

这个步骤可以过滤大量不稳定匹配点，避免错误匹配影响单应性矩阵。

### 6.4 RANSAC 和单应性矩阵

通过匹配点对估计单应性矩阵，用于把待检图映射到模板坐标系。

RANSAC 的作用是：

- 在匹配点中自动剔除离群点。
- 使用内点估计较稳定的透视变换。
- 给出内点比例，作为配准质量指标。

若单应性矩阵不可用，系统会终止当前任务，而不是继续输出不可信差异。

### 6.5 ORB 算子说明

ORB 是另一种常见特征点算法，特点是速度快、描述子为二进制，通常可配合 Hamming 距离进行匹配。它适合对性能敏感、图像质量较稳定的场景。

本项目当前默认未启用 ORB。原因是 MVP 验收约束要求使用 SIFT + FLANN + RANSAC，且扫描文档中存在缩放、旋转、模糊和纸张形变时，SIFT 通常更稳。

后续如果需要引入 ORB，建议作为可关闭的实验配置，而不是替换默认路径。推荐对照项包括：

- 配准成功率。
- 匹配点数量。
- RANSAC 内点比例。
- 最终误报数量。
- 处理耗时。

## 7. 二值化与滤波

### 7.1 二值图定义

系统内部二值图约定为：

```text
背景 = 255
前景文字 = 0
```

后续 XOR 差分基于这个约定执行。

### 7.2 双边滤波

待检图在二值化前先做双边滤波。双边滤波可以在保留文字边缘的同时平滑背景噪声。

默认参数：

```yaml
bilateral_diameter: 9
bilateral_sigma_color: 75
bilateral_sigma_space: 75
```

这一步主要应对扫描背景不均匀、纸张纹理和轻微颗粒。

### 7.3 中值滤波

本轮新增了中值滤波配置：

```yaml
median_blur_kernel: 3
```

中值滤波对孤立椒盐噪声更有效，可以减少扫描产生的小颗粒被当作文字差异。

### 7.4 自适应二值化

待检图使用高斯自适应阈值：

```yaml
adaptive_block_size: 21
adaptive_c: 10
```

自适应二值化会根据局部邻域计算阈值，比全局阈值更适合扫描件。扫描图可能存在局部阴影、页面边缘发暗、纸张亮度不一致等问题，全局 Otsu 阈值容易在这些情况下产生过多噪点。

模板图当前仍使用 Otsu 二值化。模板通常来自电子 PDF，背景更干净，使用全局阈值足够稳定。

## 8. XOR 差分

配准并二值化后，系统对两张同尺寸二值图执行逐像素 XOR：

```text
diff = scan_binary XOR template_binary
```

若某个像素在两图中的前景/背景状态不同，则输出差分像素。

XOR 的优点是简单、可解释、坐标稳定；缺点是对细微配准误差非常敏感。后续形态学处理和残影过滤主要就是为了解决这一问题。

## 9. 形态学去噪滤波

### 9.1 小连通域过滤

本轮新增 `min_noise_component_area`，在形态学处理前先移除面积过小的连通域。

```yaml
min_noise_component_area: 12
```

这一步用于过滤扫描颗粒和孤立噪点，避免后续膨胀时把小噪点放大。

### 9.2 开运算

开运算等价于先腐蚀再膨胀，主要用于去掉小的毛刺和细碎噪声。

```yaml
open_kernel: [3, 3]
morph_iterations_open: 1
```

### 9.3 闭运算

闭运算等价于先膨胀再腐蚀，主要用于连接同一字符或同一小区域内部的断裂差异。

```yaml
close_kernel: [3, 3]
morph_iterations_close: 1
```

### 9.4 膨胀

膨胀用于把同一个文字修改附近的差异像素合并成更容易复核的区域。

```yaml
dilate_kernel: [15, 10]
morph_iterations_dilate: 1
```

膨胀核不能过小，否则一个真实文字修改会被拆成大量小框；也不能过大，否则相邻差异会被合并成过大的框。

## 10. 区域提取与排序

形态学处理后，系统使用 OpenCV 轮廓提取差异区域。区域过滤和排序规则包括：

1. 面积小于 `min_diff_area` 的区域忽略。
2. 坐标裁剪到图像范围内。
3. 按 `y、x、面积降序` 稳定排序。
4. 排序后重新编号，保证输出顺序可重复。

默认：

```yaml
min_diff_area: 200
```

输出坐标使用模板坐标系，左上角为原点，单位为像素。

## 11. 文字残影可视化

本轮报告图改为文字残影风格，便于人工观察差异来源。

颜色含义：

- 红色：模板中存在、待检文档中缺失的笔画。
- 青色：待检文档中存在、模板中不存在的笔画。
- 浅白色：两份文档重合的笔画。
- 紫色背景：无正文笔画区域。

此外，系统会在残影图上绘制红色 box 和序号，与 JSON 坐标对应。

## 12. 误判过滤策略

在实际样本中，单纯依赖 XOR 和形态学仍会出现误判。主要误判来源包括：

- 文字边缘亚像素偏移造成红/青残影。
- 签名空白线、表格线被识别为长条差异。
- 页面底部或空白区出现孤立噪点。
- 整行文字因轻微配准误差被膨胀成大框。

为此，本轮在区域提取后增加了多类后过滤规则。

### 12.1 局部相似度过滤

对每个候选区域，取其周围小窗口，分别提取模板和待检二值前景。然后在小范围内平移模板前景，计算最大 IoU。

若小范围平移后两者仍高度相似，说明该区域更可能是配准残影，而不是真实文字改动。

核心配置：

```yaml
local_similarity_filter: true
local_similarity_iou_threshold: 0.62
local_similarity_padding: 8
local_similarity_search_radius: 4
```

调参依据：

- 在 `document2` 中，真实小差异的局部 IoU 最高约为 `0.6115`。
- 若阈值压到 `0.60`，会误删真实差异。
- 因此默认设置为 `0.62`，在保留真实差异前提下降低误报。

### 12.2 长横线残影过滤

签名栏、下划线、空白横线等区域容易形成细长框。

配置：

```yaml
horizontal_residual_min_aspect: 12.0
horizontal_residual_max_height: 20
```

满足宽高比极高且高度较小的区域会被过滤。

### 12.3 短横向残影过滤

`document1` 中出现过短横向残影，不满足长横线条件，但仍明显是局部文字边缘错位。

配置：

```yaml
short_horizontal_residual_min_aspect: 2.5
short_horizontal_residual_max_height: 20
short_horizontal_residual_min_iou: 0.55
```

该规则要求区域呈横向，并且局部相似度达到一定水平，避免误删真实新增短文本。

### 12.4 稀疏小噪点过滤

空白区或局部前景密度很低的位置，如果出现小框，大概率是噪点或残影。

配置：

```yaml
sparse_residual_max_area: 400
sparse_residual_max_density: 0.04
small_residual_max_area: 220
small_residual_max_density: 0.12
residual_filter_min_area: 200
residual_density_padding: 40
```

这里单独设置 `residual_filter_min_area`，避免在某些测试或低阈值场景下把真实的小点状差异直接删掉。

### 12.5 宽文本行残影过滤

`document2` 底部曾出现整行文字被框住的误判。这类区域面积很大、横向宽高比高、局部文字相似度不低，通常是整行文字边缘残影被膨胀合并造成。

配置：

```yaml
wide_text_residual_min_area: 5000
wide_text_residual_min_aspect: 3.0
wide_text_residual_min_iou: 0.30
```

调试结果：

- `document2` 中这类行级误判被过滤后，总框数从 87 降到 78。
- 已确认的 7 个真实差异坐标仍保留。
- `document1` 中 15 个真实框保持不变。

## 13. 样本调试记录

### 13.1 document2 第一轮

命令：

```powershell
.\.venv\Scripts\python.exe scripts\compare.py test_pdf\document2_v3.pdf test_pdf\document2_v1.pdf --visual artifacts\result.png --ghost artifacts\heatmap.png
```

用户确认真实差异为旧序号：

```text
10、15、28、55、162、168、248
```

调试结论：

- 原始结果中大量误差来自字符边缘残影。
- 简单提高面积阈值会误删真实小差异。
- 引入局部相似度过滤后，在保留上述真实差异坐标的前提下降低误报。

### 13.2 document1

用户确认误判为旧序号：

```text
4、9、18、19、20、21、22、23
```

调试结论：

- `19、21` 是签名空白横线残影。
- `22、23` 位于页面底部空白区，是孤立噪点。
- `4、9` 是正文附近小残影。
- `18` 是短横向残影。

处理后：

```text
23 个框 -> 15 个框
```

### 13.3 document2 第二轮

用户指出底部类似行级框为误判：

```text
71、79、80、81、82、83、85、86
```

说明：最后一行文字只有“裁定书”前面的空格是真实差异，其他行级大框为误判。

处理后：

```text
87 个框 -> 78 个框
```

已知 7 个真实差异坐标仍保留。

## 14. 配置集中化

本轮新增或调整的关键配置集中在 `configs/default.yaml` 和 `PixelDiffConfig` 中，避免阈值散落在算法代码里。

新增配置包括：

```yaml
median_blur_kernel: 3
min_noise_component_area: 12
local_similarity_filter: true
local_similarity_iou_threshold: 0.62
local_similarity_padding: 8
local_similarity_search_radius: 4
horizontal_residual_min_aspect: 12.0
horizontal_residual_max_height: 20
short_horizontal_residual_min_aspect: 2.5
short_horizontal_residual_max_height: 20
short_horizontal_residual_min_iou: 0.55
wide_text_residual_min_area: 5000
wide_text_residual_min_aspect: 3.0
wide_text_residual_min_iou: 0.30
sparse_residual_max_area: 400
sparse_residual_max_density: 0.04
small_residual_max_area: 220
small_residual_max_density: 0.12
residual_filter_min_area: 200
residual_density_padding: 40
```

这些阈值都可以在后续样本评估中继续调整。

## 15. 验证方式

每次调整误判过滤规则后，都执行以下验证：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src scripts
```

最近一次验证结果：

```text
pytest: 71 passed, 3 skipped
ruff: All checks passed
mypy: Success
```

同时人工复核：

- `document1` 保留 15 个真实框。
- `document2` 已知 7 个真实差异坐标全部保留。
- `document2` 底部行级大框误判被过滤。

## 16. 当前局限与后续方向

当前流程仍属于 MVP 的像素级筛查方案，存在以下局限：

1. 对非线性纸张形变仍不够稳，当前只做全局单应性配准。
2. 对黑色手写签名、黑色印章无法稳定过滤。
3. 对真实差异与残影极其相似的情况，仍需要人工裁决。
4. 当前未做 OCR，不输出修改前后文本。
5. ORB 尚未作为默认或实验路径接入。

后续可考虑：

- 局部网格或薄板样条配准，处理纸张非线性变形。
- 自动估计笔画宽度，补偿打印粗细差异。
- ROI 忽略区配置，跳过签章栏、页眉页脚等高噪声区域。
- 仅对候选差异区域执行 OCR，辅助人工复核。
- 引入 ORB 作为实验性快速配准选项，并与 SIFT 做同数据集对照。

## 17. 结论

本轮改动的核心思路是：

```text
先保证输入可处理
再保证页面和报告可复核
然后基于真实样本持续降低误判
最后把经验沉淀为可配置规则和自动化测试
```

当前系统仍坚持“机器筛查、人工裁决”的定位。算法负责把疑似差异收敛到更可复核的范围，最终判断仍由人工完成。
