import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from starlette.requests import Request

from pixel_diff_api.app import create_app
from pixel_diff_api.settings import ApiSettings
from pixel_diff_api.task_service import CompareTask


class FakeTaskManager:
    def __init__(self, task: CompareTask) -> None:
        self.task = task

    def get(self, task_id: str) -> CompareTask | None:
        return self.task if task_id == self.task.task_id else None


def _application(tmp_path: Path, status: str = "completed") -> FastAPI:
    output_dir = tmp_path / "run"
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "page_0001_original.jpg").write_bytes(b"jpg-data")
    (report_dir / "page_0001_candidate.jpg").write_bytes(b"jpg-data")
    (report_dir / "page_0001_heatmap.png").write_bytes(b"png-data")
    result_path = output_dir / "diff_result.json"
    result_path.write_text(
        json.dumps(
            {
                "total_pages": 1,
                "total_regions": 1,
                "differences": [{"page": 1, "id": 1, "change_label": "修改"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    task = CompareTask(
        task_id="task-1",
        status=status,  # type: ignore[arg-type]
        output_dir=str(output_dir),
        result_json=str(result_path),
        total_pages=1,
        difference_count=1,
    )
    settings = ApiSettings(project_root=tmp_path, task_root=tmp_path / "tasks")
    return create_app(settings, manager=FakeTaskManager(task))  # type: ignore[arg-type]


def _endpoint(application: FastAPI, path: str):
    return next(route.endpoint for route in application.routes if route.path == path)


def _json(response: Response) -> dict[str, object]:
    return json.loads(bytes(response.body).decode("utf-8"))


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "server": ("testserver", 80),
            "path": path,
            "root_path": "",
            "query_string": b"",
            "headers": [],
        }
    )


def test_completed_task_exposes_view_metadata_and_viewer(tmp_path: Path) -> None:
    application = _application(tmp_path)
    metadata = _endpoint(application, "/api/pixel/compare/tasks/{task_id}/view")(
        "task-1", _request("/api/pixel/compare/tasks/task-1/view")
    )
    viewer = _endpoint(application, "/api/pixel/compare/tasks/{task_id}/viewer")("task-1")

    assert metadata.status_code == 302
    assert metadata.headers["location"].endswith("/viewer")
    assert viewer.status_code == 200
    assert viewer.headers["content-type"].startswith("text/html")
    viewer_text = bytes(viewer.body).decode("utf-8")
    assert 'id="template-panel"' in viewer_text
    assert 'id="comparison-panel"' in viewer_text
    assert 'id="difference-panel"' in viewer_text


def test_processing_task_view_metadata_returns_202(tmp_path: Path) -> None:
    application = _application(tmp_path, status="running")
    response = _endpoint(application, "/api/pixel/compare/tasks/{task_id}/view")(
        "task-1", _request("/api/pixel/compare/tasks/task-1/view")
    )

    assert response.status_code == 200
    assert _json(response)["data"]["compare_view_url"] is None


def test_page_images_are_displayed_inline(tmp_path: Path) -> None:
    application = _application(tmp_path)
    endpoint = _endpoint(
        application, "/api/pixel/compare/tasks/{task_id}/pages/{page}/{image_type}"
    )

    expected_type = {"template": "image/jpeg", "candidate": "image/jpeg", "diff": "image/png"}
    for image_type in ("template", "candidate", "diff"):
        response = endpoint("task-1", 1, image_type)
        assert response.headers["content-type"] == expected_type[image_type]
        assert response.headers["content-disposition"].startswith("inline;")


def test_page_image_validation(tmp_path: Path) -> None:
    application = _application(tmp_path)
    endpoint = _endpoint(
        application, "/api/pixel/compare/tasks/{task_id}/pages/{page}/{image_type}"
    )

    for page, image_type, expected in ((0, "diff", 400), (1, "unknown", 400), (2, "diff", 404)):
        try:
            endpoint("task-1", page, image_type)
        except HTTPException as exc:
            assert exc.status_code == expected
        else:
            raise AssertionError("expected HTTPException")


def test_raw_result_contains_view_url(tmp_path: Path) -> None:
    application = _application(tmp_path)
    response = _endpoint(application, "/api/v1/compare/tasks/{task_id}/result")(
        "task-1", _request("/api/v1/compare/tasks/task-1/result")
    )

    assert response.status_code == 200
    data = _json(response)["data"]
    assert str(data["results"]["compare_view_url"]).endswith("/viewer")
