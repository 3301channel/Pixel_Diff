from pixel_diff_api.task_service import CompareTask
from pixel_diff_api.viewer import render_compare_viewer


def test_render_compare_viewer_contains_three_columns_and_page_controls() -> None:
    task = CompareTask(
        task_id="task-123",
        status="completed",
        file_name_a="template.pdf",
        file_name_b="candidate.pdf",
        difference_count=2,
        total_pages=3,
    )
    payload = {
        "total_pages": 3,
        "total_regions": 2,
        "differences": [
            {
                "page": 1,
                "id": 1,
                "change_label": "修改",
                "risk_level": "HIGH",
                "x": 10,
                "y": 20,
                "width": 30,
                "height": 40,
                "area": 1200,
                "template_text": "合同金额",
            }
        ],
    }

    html = render_compare_viewer(task, payload)

    assert 'id="template-panel"' in html
    assert 'id="comparison-panel"' in html
    assert 'id="difference-panel"' in html
    assert 'id="previous-page"' in html
    assert 'id="next-page"' in html
    assert 'id="view-mode"' in html
    assert "/api/pixel/compare/tasks/task-123/pages/" in html
    assert "合同金额" in html


def test_render_compare_viewer_escapes_task_controlled_text() -> None:
    task = CompareTask(
        task_id="task-safe",
        status="completed",
        file_name_a="<script>alert(1)</script>.pdf",
        file_name_b="candidate.pdf",
        total_pages=1,
    )

    html = render_compare_viewer(task, {"total_pages": 1, "differences": []})

    assert "<script>alert(1)</script>.pdf" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.pdf" in html


def test_render_compare_viewer_reads_regions_from_real_result_json() -> None:
    task = CompareTask(
        task_id="task-regions",
        status="completed",
        total_pages=2,
    )
    payload = {
        "total_pages": 2,
        "total_regions": 1,
        "regions": [
            {
                "page": 2,
                "id": 7,
                "x": 120,
                "y": 240,
                "width": 36,
                "height": 42,
                "change_type": "modified",
                "change_label": "修改",
                "risk_level": "MEDIUM",
            }
        ],
    }

    html = render_compare_viewer(task, payload)

    assert "result.regions" in html
    assert 'Array.isArray(result.differences)' in html
    assert 'Number(item.page) === currentPage' in html


def test_render_compare_viewer_shows_prominent_similarity() -> None:
    task = CompareTask(task_id="task-similarity", status="completed")

    html = render_compare_viewer(
        task,
        {
            "total_pages": 1,
            "total_regions": 1,
            "difference_rate": 0.000795,
            "regions": [],
        },
    )

    assert 'id="similarity-card"' in html
    assert 'id="similarity-value"' in html
    assert "99.92%" in html
    assert "整体相似度" in html
    assert "基于疑似差异像素面积计算" in html
    assert "similarity-high" in html


def test_render_compare_viewer_clamps_similarity_to_valid_percentage() -> None:
    task = CompareTask(task_id="task-clamped", status="completed")

    html = render_compare_viewer(
        task,
        {"total_pages": 1, "difference_rate": 1.5, "regions": []},
    )

    assert "0.00%" in html
    assert "similarity-low" in html


def test_render_compare_viewer_handles_missing_similarity() -> None:
    task = CompareTask(task_id="task-no-similarity", status="completed")

    html = render_compare_viewer(task, {"total_pages": 1, "regions": []})

    assert 'id="similarity-value"' in html
    assert ">--</strong>" in html
    assert "similarity-unknown" in html
