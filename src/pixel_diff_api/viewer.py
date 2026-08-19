"""Self-contained browser viewer for completed comparison tasks."""

from __future__ import annotations

import html
import json
import math
from urllib.parse import quote

from pixel_diff_api.task_service import CompareTask


def render_compare_viewer(task: CompareTask, payload: dict[str, object]) -> str:
    """Render a three-column comparison page for one completed task."""

    safe_task_id = quote(task.task_id, safe="")
    base_path = f"/api/pixel/compare/tasks/{safe_task_id}/pages/"
    result_json = _safe_json(payload)
    template_name = html.escape(task.file_name_a or "模板文档")
    candidate_name = html.escape(task.file_name_b or "待检测文档")
    total_pages = int(payload.get("total_pages") or task.total_pages or 1)
    difference_count = int(payload.get("total_regions") or task.difference_count or 0)
    similarity_value, similarity_class = _similarity_display(payload)
    elapsed_display = _elapsed_display(payload)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>文档差异查看器</title>
  <style>
    :root {{
      color-scheme: light;
      --border: #d8dee8;
      --surface: #ffffff;
      --muted: #667085;
      --primary: #1677ff;
      --danger: #e5484d;
      --background: #f4f6f9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-width: 1080px;
      background: var(--background);
      color: #1f2937;
      font: 14px/1.5 "Microsoft YaHei", "PingFang SC", sans-serif;
    }}
    header {{
      min-height: 78px;
      display: flex;
      align-items: center;
      gap: 18px;
      padding: 0 20px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
    }}
    header h1 {{ margin: 0; font-size: 19px; }}
    .meta {{ color: var(--muted); white-space: nowrap; }}
    .similarity-card {{
      min-width: 150px;
      padding: 7px 12px;
      border: 2px solid currentColor;
      border-radius: 9px;
      background: #f8fafc;
      line-height: 1.15;
      text-align: center;
    }}
    .similarity-value {{ display: block; font-size: 27px; font-weight: 800; }}
    .similarity-label {{ display: block; margin-top: 2px; font-size: 12px; font-weight: 700; }}
    .similarity-high {{ color: #16803c; background: #f0fdf4; }}
    .similarity-medium {{ color: #b45309; background: #fffbeb; }}
    .similarity-low {{ color: #d92d20; background: #fff5f3; }}
    .similarity-unknown {{ color: var(--muted); }}
    .similarity-help {{
      white-space: nowrap;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
    }}
    .controls {{ margin-left: auto; display: flex; align-items: center; gap: 8px; }}
    button, select {{
      min-height: 34px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      padding: 5px 11px;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--primary); color: var(--primary); }}
    button:disabled {{ cursor: not-allowed; opacity: .45; }}
    main {{
      height: calc(100vh - 78px);
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr) 360px;
      grid-template-rows: 1fr;
      gap: 1px;
      background: var(--border);
      overflow: hidden;
    }}
    .panel {{
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      background: var(--surface);
    }}
    .panel-title {{
      height: 46px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
      border-bottom: 1px solid var(--border);
      font-weight: 600;
    }}
    .document-scroll {{
      flex: 1;
      overflow: auto;
      padding: 14px;
      background: #eef1f5;
    }}
    .document-scroll img {{
      display: block;
      width: 100%;
      height: auto;
      background: white;
      box-shadow: 0 2px 10px rgb(16 24 40 / 12%);
      transform-origin: top left;
    }}
    .image-stage {{
      position: relative;
      display: inline-block;
      width: 100%;
    }}
    .highlight-layer {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      overflow: visible;
    }}
    .highlight-box {{
      position: absolute;
      border: 2px solid #e5484d;
      background: rgb(229 72 77 / 16%);
      box-sizing: border-box;
      border-radius: 2px;
    }}
    .highlight-box .highlight-label {{
      position: absolute;
      bottom: 100%;
      left: -2px;
      margin-bottom: 3px;
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 11px;
      font-weight: 600;
      line-height: 1.5;
      color: #fff;
      background: #e5484d;
      white-space: nowrap;
      pointer-events: none;
    }}
    .difference-list {{ flex: 1 1 0; min-height: 0; max-height: 100%; overflow-x: hidden; overflow-y: auto; padding: 12px; }}
    .difference-card {{
      margin-bottom: 10px;
      border: 1px solid var(--border);
      border-left: 4px solid var(--danger);
      border-radius: 6px;
      padding: 10px 12px;
      background: #fff;
    }}
    .difference-card.high {{ border-left-color: #e5484d; }}
    .difference-card.medium {{ border-left-color: #f59e0b; }}
    .difference-card.low {{ border-left-color: #22c55e; }}
    .difference-card h3 {{
      margin: 0 0 8px;
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
    }}
    .difference-card .risk-badge {{
      display: inline-block;
      padding: 1px 7px;
      border-radius: 4px;
      font-size: 11px;
      color: #fff;
    }}
    .risk-badge.high {{ background: #e5484d; }}
    .risk-badge.medium {{ background: #f59e0b; color: #1f2937; }}
    .risk-badge.low {{ background: #22c55e; }}
    .difference-card dl {{
      display: grid;
      grid-template-columns: 72px 1fr;
      gap: 2px 8px;
      margin: 0 0 6px;
      font-size: 12px;
    }}
    .difference-card dt {{ margin: 0; color: var(--muted); }}
    .difference-card dd {{ margin: 0; color: #344054; overflow-wrap: anywhere; }}
    .difference-card .extra {{
      margin-top: 2px;
      padding-top: 5px;
      border-top: 1px solid var(--border);
      color: var(--muted);
      font-size: 11px;
    }}
    .empty {{ padding: 36px 12px; color: var(--muted); text-align: center; }}
    .zoom {{ display: flex; align-items: center; gap: 6px; font-weight: 400; }}
    .zoom input {{ width: 90px; }}
  </style>
</head>
<body>
  <header>
    <h1>文档差异查看器</h1>
    <span class="meta">任务：{html.escape(task.task_id)}</span>
    <span class="meta">差异：{difference_count}</span>
    <span class="meta">耗时：{elapsed_display}</span>
    <div
      id="similarity-card"
      class="similarity-card {similarity_class}"
      title="基于疑似差异像素面积计算，不代表语义一致性"
    >
      <strong id="similarity-value" class="similarity-value">{similarity_value}</strong>
      <span class="similarity-label">整体相似度</span>
    </div>
    <span class="similarity-help">
      基于疑似差异像素面积计算，不代表语义一致性
    </span>
    <div class="controls">
      <button id="previous-page" type="button">上一页</button>
      <span>第 <strong id="current-page">1</strong> / {total_pages} 页</span>
      <button id="next-page" type="button">下一页</button>
        <select id="view-mode" aria-label="中栏显示模式">
          <option value="diff">差异残影图</option>
          <option value="candidate" selected>待检测原图</option>
        </select>
    </div>
  </header>
  <main>
    <section class="panel" id="template-panel">
      <div class="panel-title">
        <span>模板：{template_name}</span>
        <label class="zoom">缩放
          <input id="zoom-range" type="range" min="60" max="180" value="100">
        </label>
      </div>
      <div class="document-scroll" id="template-scroll">
        <div class="image-stage">
          <img id="template-image" alt="模板页">
          <div class="highlight-layer" id="template-highlight"></div>
        </div>
      </div>
    </section>
    <section class="panel" id="comparison-panel">
      <div class="panel-title">
        <span id="comparison-title">差异残影图</span>
        <span>{candidate_name}</span>
      </div>
      <div class="document-scroll" id="comparison-scroll">
        <div class="image-stage">
          <img id="comparison-image" alt="对比页">
          <div class="highlight-layer" id="comparison-highlight"></div>
        </div>
      </div>
    </section>
    <aside class="panel" id="difference-panel">
      <div class="panel-title">
        <span>当前页差异</span>
        <span id="page-difference-count">0</span>
      </div>
      <div class="difference-list" id="difference-list"></div>
    </aside>
  </main>
  <script>
    const result = {result_json};
    const basePath = {json.dumps(base_path)};
    const totalPages = {total_pages};
    const allDifferences = Array.isArray(result.regions) ? result.regions
      : Array.isArray(result.differences) ? result.differences : [];
    let currentPage = 1;
    let syncingScroll = false;

    const templateImage = document.getElementById("template-image");
    const comparisonImage = document.getElementById("comparison-image");
    const templateScroll = document.getElementById("template-scroll");
    const comparisonScroll = document.getElementById("comparison-scroll");
    const templateHighlight = document.getElementById("template-highlight");
    const comparisonHighlight = document.getElementById("comparison-highlight");
    const viewMode = document.getElementById("view-mode");
    const differenceList = document.getElementById("difference-list");
    let hoveringId = null;

    function text(value, fallback = "-") {{
      return value === null || value === undefined || value === "" ? fallback : String(value);
    }}

    function addDetail(list, label, value) {{
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = label;
      dd.textContent = text(value);
      list.append(dt, dd);
    }}

    function renderDifferences() {{
      const items = allDifferences.filter(item => Number(item.page) === currentPage);
      document.getElementById("page-difference-count").textContent = String(items.length);
      differenceList.replaceChildren();
      if (!items.length) {{
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "当前页没有差异";
        differenceList.appendChild(empty);
        return;
      }}
      for (const item of items) {{
        const risk = String(item.risk_level || "").toLowerCase();
        const card = document.createElement("article");
        const knownRisk = risk === "high" || risk === "medium" || risk === "low";
        card.className = "difference-card " + (knownRisk ? risk : "");
        const title = document.createElement("h3");
        const badge = document.createElement("span");
        badge.className = "risk-badge " + (knownRisk ? risk : "");
        badge.textContent = text(item.risk_level);
        title.append(
          document.createTextNode(
            "#" + text(item.id) + " " +
            text(item.change_label, text(item.change_type, "差异"))
          ),
          badge
        );
        const details = document.createElement("dl");
        // 文字 region（template_text 非空）向右下扩展，吸收签名行尾部追加字符等
        // 配准后被吸收到文字 region 内、但实际差异像素超出 OCR 框的小区域。
        const isText = text(item.template_text).length > 0;
        const padX = isText ? 50 : 0;
        const padY = isText ? 30 : 0;
        addDetail(details, "原文", String(item.template_text || "").length > 40
          ? String(item.template_text).substring(0, 40) + "…"
          : text(item.template_text));
        card.append(title, details);
        card.addEventListener("mouseenter", () => showHighlight(item, padX, padY));
        card.addEventListener("mouseleave", () => {{
          hoveringId = null;
          clearHighlight();
        }});
        differenceList.appendChild(card);
      }}
    }}

    // 悬停高亮：鼠标停在差异卡片上时，在左右两张图上框出差异区域。
    // 差异坐标为模板坐标系（= 残影图 natural 尺寸），按各图显示比例换算。
    function getBaseSize() {{
      return new Promise((resolve) => {{
        const img = new Image();
        img.onload = () => resolve({{ w: img.naturalWidth, h: img.naturalHeight }});
        img.onerror = () => resolve(null);
        img.src = `${{basePath}}${{currentPage}}/diff`;
      }});
    }}

    function highlightRectOn(img, baseW, baseH, item, padX = 0, padY = 0) {{
      if (!img || !img.naturalWidth || !baseW || !baseH) return null;
      const stage = img.parentElement;
      const sx = stage.clientWidth / baseW;
      const sy = stage.clientHeight / baseH;
      return {{
        left: Math.round(item.x * sx) + "px",
        top: Math.round(item.y * sy) + "px",
        width: Math.max(2, Math.round((item.width + padX) * sx)) + "px",
        height: Math.max(2, Math.round((item.height + padY) * sy)) + "px",
      }};
    }}

    function clearHighlight() {{
      templateHighlight.replaceChildren();
      comparisonHighlight.replaceChildren();
    }}

    function showHighlight(item, padX = 0, padY = 0) {{
      const id = item.id;
      hoveringId = id;
      clearHighlight();
      getBaseSize().then((base) => {{
        if (hoveringId !== id || !base) return;
        const boxes = [
          highlightRectOn(templateImage, base.w, base.h, item, padX, padY),
          highlightRectOn(comparisonImage, base.w, base.h, item, padX, padY),
        ];
        const layers = [templateHighlight, comparisonHighlight];
        for (let k = 0; k < 2; k++) {{
          if (!boxes[k]) continue;
          const b = document.createElement("div");
          b.className = "highlight-box";
          b.style.left = boxes[k].left;
          b.style.top = boxes[k].top;
          b.style.width = boxes[k].width;
          b.style.height = boxes[k].height;
          // 高亮框上方标注差异序号 + 差异种类
          const label = document.createElement("span");
          label.className = "highlight-label";
          label.textContent =
            "#" + text(item.id) + " " + text(item.change_label, text(item.change_type, "差异"));
          b.appendChild(label);
          layers[k].appendChild(b);
        }}
      }});
    }}

    // 预加载相邻页图片：翻页前预取前后页，命中浏览器缓存后翻页秒开，
    // 避免两张图加载时机不同导致的左右不同步。
    const imageCache = new Set();
    function preload(url) {{
      if (imageCache.has(url)) return;
      imageCache.add(url);
      const img = new Image();
      img.src = url;
    }}
    function preloadAround(page) {{
      for (let p = page - 1; p <= page + 1; p++) {{
        if (p < 1 || p > totalPages) continue;
        preload(`${{basePath}}${{p}}/template`);
        preload(`${{basePath}}${{p}}/candidate`);
        preload(`${{basePath}}${{p}}/diff`);
      }}
    }}

    function renderPage() {{
      const mode = viewMode.value;
      hoveringId = null;
      clearHighlight();
      templateScroll.scrollTo(0, 0);
      comparisonScroll.scrollTo(0, 0);
      templateImage.src = `${{basePath}}${{currentPage}}/template`;
      comparisonImage.src = `${{basePath}}${{currentPage}}/${{mode}}`;
      document.getElementById("comparison-title").textContent =
        mode === "diff" ? "差异残影图" : "待检测原图";
      document.getElementById("current-page").textContent = String(currentPage);
      document.getElementById("previous-page").disabled = currentPage <= 1;
      document.getElementById("next-page").disabled = currentPage >= totalPages;
      renderDifferences();
      preloadAround(currentPage);
    }}

    function synchronizeScroll(source, target) {{
      source.addEventListener("scroll", () => {{
        if (syncingScroll) return;
        syncingScroll = true;
        const verticalRange = Math.max(1, source.scrollHeight - source.clientHeight);
        const horizontalRange = Math.max(1, source.scrollWidth - source.clientWidth);
        target.scrollTop =
          source.scrollTop / verticalRange * Math.max(0, target.scrollHeight - target.clientHeight);
        target.scrollLeft =
          source.scrollLeft / horizontalRange *
          Math.max(0, target.scrollWidth - target.clientWidth);
        requestAnimationFrame(() => {{ syncingScroll = false; }});
      }});
    }}

    document.getElementById("previous-page").addEventListener("click", () => {{
      currentPage = Math.max(1, currentPage - 1);
      renderPage();
    }});
    document.getElementById("next-page").addEventListener("click", () => {{
      currentPage = Math.min(totalPages, currentPage + 1);
      renderPage();
    }});
    viewMode.addEventListener("change", renderPage);
    document.getElementById("zoom-range").addEventListener("input", event => {{
      const width = `${{event.target.value}}%`;
      // 缩放图片所在的 stage（而非 img），让高亮层与图片处于同一坐标系：
      // 图片是 stage 的 100%，stage 一起缩放后二者渲染尺寸始终一致，
      // 高亮框（基于 stage.clientWidth 换算）便不会因缩放而偏移。
      templateImage.parentElement.style.width = width;
      comparisonImage.parentElement.style.width = width;
    }});
    synchronizeScroll(templateScroll, comparisonScroll);
    synchronizeScroll(comparisonScroll, templateScroll);
    renderPage();
  </script>
</body>
</html>"""


def _safe_json(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        serialized.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _similarity_display(payload: dict[str, object]) -> tuple[str, str]:
    raw_rate = payload.get("difference_rate")
    if isinstance(raw_rate, bool) or not isinstance(raw_rate, (int, float)):
        return "--", "similarity-unknown"
    rate = float(raw_rate)
    if not math.isfinite(rate):
        return "--", "similarity-unknown"
    similarity = min(100.0, max(0.0, (1.0 - rate) * 100.0))
    if similarity >= 95.0:
        level = "similarity-high"
    elif similarity >= 80.0:
        level = "similarity-medium"
    else:
        level = "similarity-low"
    return f"{similarity:.2f}%", level


def _elapsed_display(payload: dict[str, object]) -> str:
    """整个任务耗时（秒），含渲染时间。

    优先用墙钟时间（wall_elapsed_ms，含渲染）；
    无墙钟时回退到算法时间合计（metrics.elapsed_ms）。
    """
    wall_ms = payload.get("wall_elapsed_ms")
    if isinstance(wall_ms, (int, float)) and wall_ms > 0:
        total_ms = float(wall_ms)
    else:
        metrics = payload.get("metrics") or {}
        total_ms = float(metrics.get("elapsed_ms") or 0)

    total_ms = max(0.0, total_ms)
    return f"{total_ms / 1000:.1f}s"


_CSS_LOADING = (
    "body{margin:0;background:#f4f6f9;color:#1f2937;"
    "font:14px/1.5 \"Microsoft YaHei\",\"PingFang SC\",sans-serif;"
    "display:flex;align-items:center;justify-content:center;height:100vh;}"
    ".card{background:#fff;border:1px solid #d8dee8;border-radius:10px;"
    "padding:36px 48px;box-shadow:0 6px 24px rgba(16,24,40,0.08);"
    "text-align:center;min-width:360px;}"
    "h1{margin:0 0 8px;font-size:18px;font-weight:600;}"
    "p.meta{margin:0 0 18px;color:#667085;font-size:12px;}"
    ".spinner{width:42px;height:42px;border:4px solid #e5e7eb;"
    "border-top-color:#1677ff;border-radius:50%;margin:14px auto;"
    "animation:spin 0.9s linear infinite;}"
    "@keyframes spin{to{transform:rotate(360deg);}}"
    ".hint{color:#667085;font-size:12px;margin-top:14px;}"
    ".error{color:#d92d20;}"
)


_POLL_SCRIPT = (
    "(async function poll(){"
    "while(true){"
    "try{"
    "const r=await fetch('/api/v1/compare/tasks/__TID__/result');"
    "const j=await r.json();"
    "const s=j.data && j.data.status;"
    "if(s==='completed'||s==='failed'){location.replace(location.href);return;}"
    "}catch(e){}"
    "await new Promise(function(r){setTimeout(r,2000);});"
    "}})();"
)


def render_loading_viewer(task: CompareTask) -> str:
    """Render a lightweight loading/error page for a non-completed task.

    While pending/running, the page polls /result every 2s and replaces itself
    once the task completes (server then serves the full result viewer).
    For a failed task, the error is shown inline without polling.
    """
    safe_id = html.escape(task.task_id)
    body = (
        '<div class="card"><h1>正在比对文档...</h1>'
        f'<p class="meta">任务 ID：{safe_id}</p>'
        '<div class="spinner"></div>'
        '<p class="hint">比对完成后将自动展示结果</p></div>'
    )
    script = _POLL_SCRIPT.replace("__TID__", safe_id)
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>文档差异查看器 - 比对中</title><style>'
        + _CSS_LOADING + '</style></head><body>' + body
        + ('<script>' + script + '</script>' if script else '')
        + '</body></html>'
    )


_CSS_FAILED = (
    "body{margin:0;background:#f4f6f9;color:#1f2937;"
    "font:14px/1.5 \"Microsoft YaHei\",\"PingFang SC\",sans-serif;}"
    "header{background:#fff;border-bottom:1px solid #d8dee8;padding:14px 20px;}"
    ".error-banner{background:#fff5f3;border:1px solid #e5484d;color:#d92d20;"
    "border-radius:8px;padding:10px 14px;font-weight:600;}"
    "p.meta{margin:6px 0 0;color:#667085;font-size:12px;}"
    "main{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#d8dee8;"
    "height:calc(100vh - 112px);}"
    ".panel{background:#fff;min-width:0;display:flex;flex-direction:column;}"
    ".panel-title{padding:10px 14px;font-weight:600;border-bottom:1px solid #d8dee8;}"
    ".scroll{flex:1;overflow:auto;padding:14px;background:#eef1f5;}"
    ".scroll img{width:100%;height:auto;background:#fff;"
    "box-shadow:0 2px 8px rgba(16,24,40,0.12);}"
)

# 与正常比对结果页共享的公共 CSS（颜色变量、header、grid 布局）
_CSS_VIEWER_COMMON = (
    ":root{color-scheme:light;"
    "--border:#d8dee8;--surface:#ffffff;--muted:#667085;"
    "--primary:#1677ff;--danger:#e5484d;--background:#f4f6f9;}"
    "*{box-sizing:border-box;}"
    "body{margin:0;min-width:1080px;background:var(--background);"
    "color:#1f2937;font:14px/1.5 \"Microsoft YaHei\",\"PingFang SC\",sans-serif;}"
    "header{min-height:78px;display:flex;align-items:center;gap:18px;"
    "padding:0 20px;background:var(--surface);border-bottom:1px solid var(--border);}"
    "header h1{margin:0;font-size:19px;}"
    ".meta{color:var(--muted);white-space:nowrap;}"
    ".mismatch-card{min-width:200px;padding:7px 12px;"
    "border:2px solid var(--danger);border-radius:9px;background:#fff5f3;"
    "line-height:1.3;text-align:center;}"
    ".mismatch-value{display:block;font-size:15px;font-weight:800;color:var(--danger);}"
    ".mismatch-label{display:block;margin-top:1px;font-size:10px;font-weight:700;color:var(--danger);}"
    "main{height:calc(100vh - 78px);display:grid;"
    "grid-template-columns:minmax(360px,1fr) minmax(360px,1fr) 360px;"
    "gap:1px;background:var(--border);}"
    ".panel{min-width:0;display:flex;flex-direction:column;background:var(--surface);}"
    ".panel-title{height:46px;display:flex;align-items:center;"
    "padding:0 14px;border-bottom:1px solid var(--border);font-weight:600;}"
    ".document-scroll{flex:1;overflow:auto;padding:14px;background:#eef1f5;}"
    ".document-scroll img{display:block;width:100%;height:auto;"
    "background:white;box-shadow:0 2px 10px rgb(16 24 40 / 12%);}"
    ".info-panel{flex:1;overflow:auto;padding:16px;}"
    ".info-card{border:2px solid #fecaca;border-radius:8px;"
    "padding:16px;background:#fff5f3;}"
    ".info-card h3{margin:0 0 10px;font-size:15px;color:var(--danger);}"
    ".info-card p{margin:6px 0;font-size:13px;color:#344054;line-height:1.6;}"
    ".info-card .task-id{font-size:11px;color:var(--muted);word-break:break-all;"
    "margin-top:12px;padding-top:10px;border-top:1px solid #fecaca;}"
    ".info-card .hint{font-size:11px;color:var(--muted);}"
)


def render_failed_viewer(task: CompareTask, base_path: str = "") -> str:
    """文档不一致时：与正常比对结果页保持一致的三栏布局。

    左栏 = 模板原图，中栏 = 待检原图，右栏 = 差异过大信息卡片。
    顶部 header 与正常比对页完全一致。
    """
    safe_id = quote(task.task_id, safe="")
    img_base = f"/api/pixel/compare/tasks/{safe_id}"
    template_name = html.escape(task.file_name_a or "原件 A（模板）")
    candidate_name = html.escape(task.file_name_b or "待检件 B（扫描件）")
    error = html.escape(task.error or "比对失败")

    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>文档差异查看器 — 文档不一致</title><style>'
        + _CSS_VIEWER_COMMON + '</style></head><body>'
        '<header>'
        '<h1>文档差异查看器</h1>'
        f'<span class="meta">任务：{html.escape(task.task_id)}</span>'
        '<div class="mismatch-card">'
        '<strong class="mismatch-value">文档不一致</strong>'
        '<span class="mismatch-label">差异过大</span>'
        '</div>'
        '</header>'
        '<main>'
        '<section class="panel">'
        f'<div class="panel-title">模板：{template_name}</div>'
        '<div class="document-scroll">'
        f'<img src="{img_base}/input_image/template" alt="模板原图">'
        '</div></section>'
        '<section class="panel">'
        f'<div class="panel-title">待检件：{candidate_name}</div>'
        '<div class="document-scroll">'
        f'<img src="{img_base}/input_image/scan" alt="待检原图">'
        '</div></section>'
        '<aside class="panel">'
        '<div class="panel-title">比对结果</div>'
        '<div class="info-panel">'
        '<div class="info-card">'
        '<h3>比对文档不一致，差异过大</h3>'
        f'<p>{error}</p>'
        '<p class="hint">请重新创建任务或联系管理员。</p>'
        f'<p class="task-id">任务 ID：{html.escape(task.task_id)}</p>'
        '</div></div></aside>'
        '</main></body></html>'
    )


def render_settings_page(workers: int, cpu_count: int) -> str:
    """渲染系统设置页：核数调整（确认生效）+ 实时内存占用。

    页面加载后轮询 ``/api/v1/system/memory`` 实时刷新内存，轮询
    ``/api/v1/settings`` 同步当前生效核数。核数修改需点击「确认生效」按钮
    才写入后端并持久化。下方预留了后续设置项的扩展位。
    """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>系统设置</title>
  <style>
    :root {{
      color-scheme: light;
      --border: #d8dee8;
      --surface: #ffffff;
      --muted: #667085;
      --primary: #1677ff;
      --danger: #e5484d;
      --background: #f4f6f9;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; min-width: 760px; background: var(--background);
      color: #1f2937; font: 14px/1.5 "Microsoft YaHei", "PingFang SC", sans-serif;
    }}
    header {{
      min-height: 64px; display: flex; align-items: center; gap: 16px;
      padding: 0 24px; background: var(--surface);
      border-bottom: 1px solid var(--border);
    }}
    header h1 {{ margin: 0; font-size: 19px; }}
    .meta {{ color: var(--muted); }}
    main {{ max-width: 860px; margin: 0 auto; padding: 24px; }}
    .card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 10px; padding: 20px 24px; margin-bottom: 18px;
      box-shadow: 0 2px 10px rgb(16 24 40 / 6%);
    }}
    .card h2 {{ margin: 0 0 6px; font-size: 16px; }}
    .desc {{ margin: 0 0 16px; color: var(--muted); font-size: 12px; }}
    .worker-row {{ display: flex; align-items: center; gap: 12px; }}
    input[type="number"] {{
      width: 92px; min-height: 36px; border: 1px solid var(--border);
      border-radius: 6px; padding: 5px 10px; font-size: 15px;
    }}
    button {{
      min-height: 36px; border: 1px solid var(--border); border-radius: 6px;
      background: #fff; padding: 6px 16px; cursor: pointer; font-size: 14px;
    }}
    button.primary {{ background: var(--primary); border-color: var(--primary); color: #fff; }}
    button.primary:hover {{ background: #0e5fd0; }}
    button:disabled {{ cursor: not-allowed; opacity: .5; }}
    .status {{ font-size: 13px; min-width: 120px; }}
    .status.ok {{ color: #16803c; }}
    .status.err {{ color: var(--danger); }}
    .hint {{ margin: 12px 0 0; color: var(--muted); font-size: 12px; }}
    .memory-bar {{
      height: 22px; background: #eef1f5; border-radius: 6px; overflow: hidden;
    }}
    .memory-fill {{
      height: 100%; width: 0%; background: linear-gradient(90deg, #1677ff, #4aa3ff);
      transition: width .4s ease;
    }}
    .memory-fill.warn {{ background: linear-gradient(90deg, #f59e0b, #fbbf24); }}
    .memory-fill.crit {{ background: linear-gradient(90deg, #e5484d, #f97066); }}
    .memory-text {{ margin: 10px 0 4px; font-size: 13px; color: #344054; }}
    .memory-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 24px; }}
    .kv {{ display: flex; justify-content: space-between; font-size: 13px; color: #344054; }}
    .kv .v {{ font-weight: 600; }}
  </style>
</head>
<body>
  <header>
    <h1>系统设置</h1>
    <span class="meta">文档比对服务</span>
  </header>
  <main>
    <section class="card">
      <h2>比对核数</h2>
      <p class="desc">多页文档并行比对使用的进程数，范围 1 ~ {cpu_count}（本机逻辑核数）。</p>
      <div class="worker-row">
        <input id="workers" type="number" min="1" max="{cpu_count}" value="{workers}">
        <span>/ {cpu_count} 核</span>
        <button id="apply-workers" class="primary" type="button">确认生效</button>
        <span id="apply-status" class="status"></span>
      </div>
      <p class="hint">当前生效核数：<strong id="current-workers">{workers}</strong></p>
    </section>

    <section class="card">
      <h2>内存占用（实时）</h2>
      <div class="memory-bar"><div id="memory-fill" class="memory-fill"></div></div>
      <p class="memory-text" id="memory-text">加载中…</p>
      <div class="memory-grid">
        <div class="kv"><span>系统总内存</span><span class="v" id="mem-total">--</span></div>
        <div class="kv"><span>系统可用内存</span><span class="v" id="mem-available">--</span></div>
        <div class="kv"><span>系统已用内存</span><span class="v" id="mem-used">--</span></div>
        <div class="kv"><span>服务进程内存</span><span class="v" id="mem-process">--</span></div>
      </div>
    </section>

    <!-- 预留：后续可在此追加更多设置项（如日志级别、算法开关、超时时间等） -->
  </main>
  <script>
    const cpuCount = {cpu_count};

    function formatBytes(bytes) {{
      if (bytes === null || bytes === undefined || bytes <= 0) return "--";
      const units = ["B", "KB", "MB", "GB", "TB"];
      let i = 0; let v = Number(bytes);
      while (v >= 1024 && i < units.length - 1) {{ v /= 1024; i++; }}
      return v.toFixed(i === 0 ? 0 : 1) + " " + units[i];
    }}

    function setStatus(text, ok) {{
      const el = document.getElementById("apply-status");
      el.textContent = text;
      el.className = "status " + (ok ? "ok" : "err");
    }}

    async function refreshMemory() {{
      try {{
        const r = await fetch("/api/v1/system/memory");
        const j = await r.json();
        const d = (j && j.data) || j || {{}};
        const percent = Number(d.percent) || 0;
        const fill = document.getElementById("memory-fill");
        fill.style.width = Math.min(100, Math.max(0, percent)) + "%";
        fill.className = "memory-fill" + (percent >= 90 ? " crit" : percent >= 75 ? " warn" : "");
        document.getElementById("memory-text").textContent =
          "系统内存使用率 " + percent.toFixed(1) + "%";
        document.getElementById("mem-total").textContent = formatBytes(d.total);
        document.getElementById("mem-available").textContent = formatBytes(d.available);
        document.getElementById("mem-used").textContent = formatBytes(d.used);
        document.getElementById("mem-process").textContent = formatBytes(d.process_rss);
      }} catch (e) {{}}
    }}

    async function refreshSettings() {{
      try {{
        const r = await fetch("/api/v1/settings");
        const j = await r.json();
        const d = (j && j.data) || j || {{}};
        if (d.report_workers != null) {{
          document.getElementById("current-workers").textContent = String(d.report_workers);
        }}
      }} catch (e) {{}}
    }}

    document.getElementById("apply-workers").addEventListener("click", async () => {{
      const n = parseInt(document.getElementById("workers").value, 10);
      if (!Number.isInteger(n) || n < 1 || n > cpuCount) {{
        setStatus("请输入 1~" + cpuCount + " 的整数", false);
        return;
      }}
      setStatus("正在保存…", true);
      try {{
        const r = await fetch("/api/v1/settings", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ report_workers: n }}),
        }});
        const j = await r.json();
        if (j && j.code === 200) {{
          document.getElementById("current-workers").textContent = String(n);
          setStatus("已生效 ✓", true);
        }} else {{
          setStatus((j && j.msg) || "保存失败", false);
        }}
      }} catch (e) {{
        setStatus("请求失败", false);
      }}
    }});

    refreshMemory();
    refreshSettings();
    setInterval(refreshMemory, 1500);
    setInterval(refreshSettings, 3000);
  </script>
</body>
</html>"""
