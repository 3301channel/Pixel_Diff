# Viewer Loading 页开发文档

> 适用范围：Pixel-Diff HTTP API（`src/pixel_diff_api/`）
> 功能：任务未完成时，展示接口返回 HTML loading 页（而非 JSON/409），完成后自动展示三栏比对结果

## 一、背景与目标

创建比对任务后，任务需要几秒到几十秒才能完成。此前前端访问展示接口时：

- 任务未完成 → 返回 `409 {"detail":"task status is running"}` 或 `{"code":200,...,"compare_view_url":null}` 这类 JSON，前端需要自行处理轮询和错误态，体验差。
- 任务完成 → 正常展示三栏结果页。

**目标**：任务未完成时，`view` / `viewer` 两个展示接口直接返回一个带 loading 动画的 HTML 页面，页面自动轮询任务状态，完成后自动刷新展示比对结果；失败时显示错误信息。前端无需再写轮询逻辑。

## 二、涉及接口

| 接口 | 说明 | 未完成（pending/running） | 已完成（completed） | 失败（failed） |
|---|---|---|---|---|
| `GET /api/pixel/compare/tasks/{task_id}/view` | 展示入口（前端一般用它拿展示地址） | `200` + HTML loading 页 | `302` → viewer 页 | HTML 错误页 |
| `GET /api/pixel/compare/tasks/{task_id}/viewer` | 三栏比对页 | `200` + HTML loading 页 | `200` + 完整三栏结果页 | HTML 错误页 |

两个接口在任务未完成时返回的是**同一个 loading/错误页**（`render_loading_viewer()`），统一体验。

## 三、状态机与页面关系

任务内部状态与对外状态的映射（`task_service.py` 的 `_map_status`）：

| 内部状态 | 对外状态（`data.status`） | HTTP 状态码 | 页面行为 |
|---|---|---|---|
| `pending` | `pending` | 202 | loading 页，轮询中 |
| `running` | `processing` | 202 | loading 页，轮询中 |
| `completed` | `completed` | 200 | 完整三栏结果页 |
| `failed` | `failed` | 200（code=50002） | 错误页（显示 `error`，不轮询） |

> 注意：日志里打印的是**内部状态**（`running`），接口返回的是**对外状态**（`processing`），二者是同一个状态。

## 四、实现细节

### 4.1 `render_loading_viewer(task)` —— `src/pixel_diff_api/viewer.py`

新增函数，返回轻量 HTML 页面：

- **failed 状态**：直接渲染错误卡片（`任务 ID + error 信息`），无轮询脚本。
- **pending / running 状态**：渲染 loading 卡片（`正在比对文档...` + CSS spinner + 任务 ID），并内嵌轮询脚本 `_POLL_SCRIPT`。

页面结构：

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>文档差异查看器 - 比对中</title>
  <style>/* _CSS_LOADING：居中卡片 + spinner 动画 */</style>
</head>
<body>
  <div class="card">
    <h1>正在比对文档...</h1>
    <p class="meta">任务 ID：{task_id}</p>
    <div class="spinner"></div>
    <p class="hint">比对完成后将自动展示结果</p>
  </div>
  <script>/* _POLL_SCRIPT：轮询 /result */</script>
</body>
</html>
```

轮询脚本逻辑（`_POLL_SCRIPT`，`__TID__` 会被替换为转义后的 task_id）：

```js
(async function poll(){
  while(true){
    try{
      const r = await fetch('/api/v1/compare/tasks/__TID__/result');
      const j = await r.json();
      const s = j.data && j.data.status;
      if (s === 'completed') { location.replace(location.href); return; }   // 完成 → 刷新，服务端渲染结果页
      if (s === 'failed') {                                                 // 失败 → 就地显示错误
        document.body.innerHTML = '<div class="card"><h1>比对失败</h1>...' + (j.data.error || '未知错误') + '...</div>';
        return;
      }
    } catch(e) {}
    await new Promise(function(r){ setTimeout(r, 2000); });                  // 每 2 秒轮询一次
  }
})();
```

关键点：

- **轮询间隔 2 秒**，直到 `completed` 或 `failed`。
- **完成时**用 `location.replace(location.href)` 重新加载当前 URL——服务端此时已能渲染完整结果页，无需前端手动跳转。
- **失败时**直接替换页面内容显示错误，停止轮询。
- task_id 注入 HTML 时经 `html.escape` 处理，防止注入。
- 页面不请求图片资源，故不会触发 `/pages/{page}/{image_type}` 的 409 检查。

### 4.2 路由分支 —— `src/pixel_diff_api/app.py`

**`get_compare_viewer`（/viewer）**：原来非 completed 直接抛 409，现改为：

```python
if task.status == "completed":
    payload = json.loads(Path(task.result_json).read_text(encoding="utf-8"))
    logger.info("serve viewer result task_id=%s pages=%s regions=%s", ...)
    return HTMLResponse(render_compare_viewer(task, payload))
# pending / running / failed → 返回 loading/错误页（前端轮询直到完成）
logger.info("serve viewer loading task_id=%s status=%s", task_id, task.status)
return HTMLResponse(render_loading_viewer(task))
```

**`get_compare_view`（/view）**：原来未完成返回 JSON（`compare_view_url: null`），现改为：

```python
if task.status == "completed":
    return RedirectResponse(f"/api/pixel/compare/tasks/{task_id}/viewer", status_code=302)
# 未完成：直接返回 HTML loading 页
logger.info("serve view loading task_id=%s status=%s", task_id, task.status)
return HTMLResponse(render_loading_viewer(task))
```

## 五、改动文件清单

| 文件 | 改动 |
|---|---|
| `src/pixel_diff_api/viewer.py` | 新增 `render_loading_viewer()`、`_CSS_LOADING`、`_POLL_SCRIPT` |
| `src/pixel_diff_api/app.py` | `get_compare_view` / `get_compare_viewer` 分支改造（移除 409）、`render_loading_viewer` 导入、加载页访问日志 |

## 六、前端调用方式

前端拿到 `data.task_id` 后，展示地址为：

```
/api/pixel/compare/tasks/{task_id}/view
```

- **iframe 嵌入**或 **location 跳转**均可：任务未完成时自动看到 loading 动画，完成自动变三栏结果页。
- 客户端模块 `ocr_hl_offline.js` 中 `getLink(id)` 即返回该地址。
- 若前端需要等待后再跳转，可先轮询 `GET /api/v1/compare/tasks/{task_id}/result` 到 `completed` 再打开链接（两种方式任选）。

## 七、验证方法

1. 启动服务：`python scripts/run_api.py --host 127.0.0.1 --port 8000`
2. 创建任务（`POST /api/v1/compare/tasks`，字段 `file_a` / `file_b`）
3. **立即**访问 `/api/pixel/compare/tasks/{task_id}/view`：
   - 期望：HTTP 200，响应为 HTML（含 `正在比对文档...`、`spinner`），**不是** JSON、不是 409
4. 等待任务完成（约 2 秒后页面自动刷新）：
   - 期望：页面变为完整三栏结果页（`文档差异查看器`、`整体相似度`、模板/残影/差异列表）
5. 接口侧验证：
   - `GET /viewer` 未完成 → 200 + loading HTML
   - 完成后 `GET /view` → 302
6. 日志验证（`GET /api/v1/logs/export?task_id={task_id}`）：
   - `serve viewer loading task_id=... status=running`（未完成访问）
   - `serve viewer result task_id=... pages=... regions=...`（完成渲染）
   - `http GET .../view -> 302`（完成时重定向）

## 八、注意事项

- **状态命名**：对外接口只出现 `pending / processing / completed / failed`；日志中出现 `running` 即 `processing`，勿混淆。
- **失败任务**：直接显示错误页，不轮询；`failed` 时 `data.error` 会带失败原因。
- **日志安全**：接口日志记录入参/出参摘要，不记录文件二进制与大段结果 JSON，避免日志膨胀。
- **task_id 注入**：渲染页面时对 task_id 做 `html.escape`，防止 XSS。
- **兼容性**：`view` 接口未完成时由"返回 JSON"改为"返回 HTML"，依赖该接口 JSON 形态的旧调用方需要同步调整（`data.compare_view_url` 字段不再返回）。
