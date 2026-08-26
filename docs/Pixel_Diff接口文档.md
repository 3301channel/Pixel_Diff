# Pixel Diff 文档对比服务接口文档

## 1. 文档信息

| 项目         | 内容                                  |
| ---------- | ----------------------------------- |
| 服务名称       | Pixel Diff 文档对比服务                   |
| API 版本     | v1                                  |
| 默认地址       | `http://127.0.0.1:8000`             |
| Swagger 文档 | `http://127.0.0.1:8000/docs`        |
| 数据格式       | JSON、`multipart/form-data`、HTML、PNG |
| 字符编码       | UTF-8                               |

## 2. 服务说明

服务采用异步任务模式：

1. 客户端上传原件 A 和待对比文件 B。
2. 服务返回 `task_id`，并在后台执行文档渲染、配准和差异检测。
3. 客户端使用 `task_id` 轮询任务状态。
4. 任务完成后，可读取结构化结果或打开三栏对比页面。
5. 任务不再需要时，可调用删除接口清理任务和产物。

支持的文件扩展名：

- `.pdf`
- `.docx`
- `.png`
- `.jpg`
- `.jpeg`

单个上传文件默认最大为 100 MB。任务默认最长执行时间为 900 秒。

## 3. 通用响应结构

正常业务响应通常采用以下结构：

```json
{
  "code": 200,
  "msg": "OK",
  "data": {}
}
```

字段说明：

| 字段     | 类型          | 说明      |
| ------ | ----------- | ------- |
| `code` | integer     | 业务状态码   |
| `msg`  | string      | 状态或错误说明 |
| `data` | object/null | 响应数据    |

### 3.1 任务状态

| 状态           | 说明         |
| ------------ | ---------- |
| `pending`    | 已进入队列，尚未开始 |
| `processing` | 正在执行对比     |
| `completed`  | 对比完成       |
| `failed`     | 对比失败       |

## 4. 创建文档对比任务

### 4.1 请求

```http
POST /api/v1/compare/tasks
Content-Type: multipart/form-data
```

请求参数：

| 参数                  | 类型      | 必填  | 默认值     | 说明               |
| ------------------- | ------- | ---:| ------- | ---------------- |
| `file_a`            | file    | 是   | -       | 原件或模板文件          |
| `file_b`            | file    | 是   | -       | 待检测文件            |
| `ignore_categories` | string  | 否   | `[]`    | 预留的忽略类别 JSON 字符串 |
| `space_recognition` | boolean | 否   | `false` | 预留的空格识别开关        |

> 注意：当前版本可以接收 `ignore_categories` 和 `space_recognition`，但尚未将其传入底层检测算法，因此这两个参数目前不会改变检测结果。

### 4.2 cURL 示例

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/compare/tasks" \
  -H "accept: application/json" \
  -F "file_a=@document2_v1.pdf" \
  -F "file_b=@document2_v3.pdf" \
  -F "ignore_categories=[]" \
  -F "space_recognition=false"
```

### 4.3 成功响应

HTTP 状态码：`202 Accepted`

```json
{
  "code": 200,
  "msg": "Task submitted successfully",
  "data": {
    "task_id": "ace029b192ef4c0db762fc5898043841",
    "status": "pending",
    "queued_ahead": 0,
    "created_at": "2026-07-29T01:40:47.063890+00:00",
    "started_at": null,
    "completed_at": null,
    "error": null,
    "file_name_a": "document2_v1.pdf",
    "file_name_b": "document2_v3.pdf",
    "status_url": "http://127.0.0.1:8000/api/v1/compare/tasks/ace029b192ef4c0db762fc5898043841/result",
    "result_url": "http://127.0.0.1:8000/api/v1/compare/tasks/ace029b192ef4c0db762fc5898043841/result"
  }
}
```

### 4.4 上传失败

HTTP 状态码：`400 Bad Request`

```json
{
  "code": 40001,
  "msg": "unsupported file type; allowed extensions: .docx, .jpeg, .jpg, .pdf, .png",
  "data": null
}
```

文件超过大小限制时，也返回业务码 `40001`。

## 5. 查询任务状态和比对结果

### 5.1 请求

```http
GET /api/v1/compare/tasks/{task_id}/result
```

路径参数：

| 参数        | 类型     | 必填  | 说明           |
| --------- | ------ | ---:| ------------ |
| `task_id` | string | 是   | 创建任务时返回的任务标识 |

建议任务未完成时每隔 2～3 秒轮询一次。

### 5.2 cURL 示例

```bash
curl "http://127.0.0.1:8000/api/v1/compare/tasks/ace029b192ef4c0db762fc5898043841/result"
```

### 5.3 处理中响应

HTTP 状态码：`202 Accepted`

```json
{
  "code": 200,
  "msg": "OK",
  "data": {
    "task_id": "ace029b192ef4c0db762fc5898043841",
    "status": "processing",
    "queued_ahead": 0,
    "created_at": "2026-07-29T01:40:47.063890+00:00",
    "started_at": "2026-07-29T01:40:47.064590+00:00",
    "completed_at": null,
    "error": null,
    "file_name_a": "document2_v1.pdf",
    "file_name_b": "document2_v3.pdf",
    "results": null
  }
}
```

### 5.4 完成响应

HTTP 状态码：`200 OK`

```json
{
  "code": 200,
  "msg": "OK",
  "data": {
    "task_id": "ace029b192ef4c0db762fc5898043841",
    "status": "completed",
    "queued_ahead": 0,
    "created_at": "2026-07-29T01:40:47.063890+00:00",
    "started_at": "2026-07-29T01:40:47.064590+00:00",
    "completed_at": "2026-07-29T01:41:28.014404+00:00",
    "error": null,
    "file_name_a": "document2_v1.pdf",
    "file_name_b": "document2_v3.pdf",
    "results": {
      "similarity": 98.7,
      "difference_count": {
        "added": 2,
        "deleted": 1,
        "modified": 4
      },
      "compare_view_url": "http://127.0.0.1:8000/api/pixel/compare/tasks/ace029b192ef4c0db762fc5898043841/viewer",
      "preview_url_a": null,
      "preview_url_b": null,
      "diff_list": [
        {
          "tag": "replace",
          "text_a": "原件区域文字",
          "text_b": "原件区域文字",
          "bbox_a": [
            {
              "char": "",
              "bbox": [120, 240, 196, 282],
              "page_idx": 0
            }
          ],
          "bbox_b": null,
          "page_idx": 0,
          "risk_level": "MEDIUM",
          "risk_reason": "large_visual_residual_without_text",
          "area": 3192.0
        }
      ]
    }
  }
}
```

`results` 字段说明：

| 字段                          | 类型      | 说明               |
| --------------------------- | ------- | ---------------- |
| `similarity`                | number  | 根据差异率换算的相似度百分比   |
| `difference_count.added`    | integer | 新增差异数量           |
| `difference_count.deleted`  | integer | 删除差异数量           |
| `difference_count.modified` | integer | 修改差异数量           |
| `compare_view_url`          | string  | 三栏结果展示页面地址       |
| `preview_url_a`             | null    | 预留字段，当前未提供独立预览地址 |
| `preview_url_b`             | null    | 预留字段，当前未提供独立预览地址 |
| `diff_list`                 | array   | 差异区域列表           |

`diff_list` 项说明：

| 字段            | 类型          | 说明                            |
| ------------- | ----------- | ----------------------------- |
| `tag`         | string      | `insert`、`delete` 或 `replace` |
| `text_a`      | string      | 原件区域文字                        |
| `text_b`      | string      | 对比件区域文字                       |
| `bbox_a`      | array       | 原件差异框及页码                      |
| `bbox_b`      | null        | 当前版本未单独返回对比件差异框               |
| `page_idx`    | integer     | 从 0 开始的页码                     |
| `risk_level`  | string/null | 风险等级                          |
| `risk_reason` | string/null | 风险判定原因                        |
| `area`        | number/null | 差异区域面积                        |

### 5.5 失败响应

当前实现中，任务执行失败时 HTTP 状态码仍为 `200 OK`，应以业务码和任务状态判断。

```json
{
  "code": 50002,
  "msg": "comparison failed",
  "data": {
    "task_id": "74a35af57a964980a446534a789403f7",
    "status": "failed",
    "queued_ahead": 0,
    "created_at": "2026-07-29T02:42:37.109673+00:00",
    "started_at": "2026-07-29T02:42:37.111255+00:00",
    "completed_at": "2026-07-29T02:42:38.014404+00:00",
    "error": "具体错误信息",
    "file_name_a": "document1_v1.pdf",
    "file_name_b": "document1_v3.pdf",
    "results": null
  }
}
```

### 5.6 任务不存在

HTTP 状态码：`404 Not Found`

```json
{
  "detail": "task not found"
}
```

## 6. 打开三栏对比页面

### 6.1 展示入口

```http
GET /api/pixel/compare/tasks/{task_id}/view
```

当任务已经完成时，接口返回 `302 Found`，并跳转到：

```text
/api/pixel/compare/tasks/{task_id}/viewer
```

当任务仍在处理中时，接口返回任务状态 JSON，不执行跳转。

浏览器会自动跟随跳转。使用 cURL 时需要添加 `-L`：

```bash
curl -L "http://127.0.0.1:8000/api/pixel/compare/tasks/ace029b192ef4c0db762fc5898043841/view"
```

### 6.2 直接打开展示页

推荐最终用户直接访问：

```text
http://127.0.0.1:8000/api/pixel/compare/tasks/{task_id}/viewer
```

页面采用三栏布局：

- 左栏：原件或模板页面；
- 中栏：差分图，可切换为待检测原图；
- 右栏：从结果 JSON 的 `regions` 中提取的当前页差异列表。

右侧差异列表按页展示差异编号、类型、风险、位置、尺寸、面积、新增/删除像素及分类信息。

## 7. 页面图像接口

该接口供三栏展示页内部调用，不在 Swagger 中展示。

```http
GET /api/pixel/compare/tasks/{task_id}/pages/{page}/{image_type}
```

路径参数：

| 参数           | 类型      | 说明                              |
| ------------ | ------- | ------------------------------- |
| `task_id`    | string  | 任务标识                            |
| `page`       | integer | 从 1 开始的页码                       |
| `image_type` | string  | `template`、`candidate` 或 `diff` |

图像类型：

| 值           | 说明      |
| ----------- | ------- |
| `template`  | 原件或模板页面 |
| `candidate` | 待检测页面   |
| `diff`      | 差异残影图   |

成功时返回 `image/png`，并以内联方式在浏览器显示。

## 8. 删除任务

### 8.1 请求

```http
DELETE /api/v1/compare/tasks/{task_id}
```

### 8.2 cURL 示例

```bash
curl -X DELETE \
  "http://127.0.0.1:8000/api/v1/compare/tasks/ace029b192ef4c0db762fc5898043841"
```

### 8.3 成功响应

HTTP 状态码：`200 OK`

```json
{
  "code": 200,
  "msg": "Task files deleted successfully",
  "data": {
    "task_id": "ace029b192ef4c0db762fc5898043841"
  }
}
```

任务处于 `pending` 或 `processing` 状态时不能删除，返回 `409 Conflict`。

## 9. 业务码与 HTTP 状态

| 场景        | HTTP 状态 | 业务码   |
| --------- | -------:| -----:|
| 创建任务成功    | 202     | 200   |
| 任务处理中     | 202     | 200   |
| 查询完成结果    | 200     | 200   |
| 上传参数或文件错误 | 400     | 40001 |
| 任务执行失败    | 200     | 50002 |
| 任务不存在     | 404     | -     |
| 删除运行中的任务  | 409     | -     |

## 10. Python 调用示例

```python
from pathlib import Path
import time

import requests


base_url = "http://127.0.0.1:8000"

with (
    Path("document2_v1.pdf").open("rb") as file_a,
    Path("document2_v3.pdf").open("rb") as file_b,
):
    response = requests.post(
        f"{base_url}/api/v1/compare/tasks",
        files={
            "file_a": ("document2_v1.pdf", file_a, "application/pdf"),
            "file_b": ("document2_v3.pdf", file_b, "application/pdf"),
        },
        data={
            "ignore_categories": "[]",
            "space_recognition": "false",
        },
        timeout=120,
    )

response.raise_for_status()
task_id = response.json()["data"]["task_id"]

while True:
    response = requests.get(
        f"{base_url}/api/v1/compare/tasks/{task_id}/result",
        timeout=30,
    )
    payload = response.json()
    status = payload.get("data", {}).get("status")

    if status == "completed":
        print(payload["data"]["results"])
        print(payload["data"]["results"]["compare_view_url"])
        break
    if status == "failed":
        raise RuntimeError(payload["data"].get("error") or payload["msg"])

    time.sleep(2)
```

## 11. 启动方式

开发环境：

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8000
```

PyInstaller 发布环境：

```bash
./run_api --host 0.0.0.0 --port 8000
```

若只允许本机访问，请将 `0.0.0.0` 改为 `127.0.0.1`。
