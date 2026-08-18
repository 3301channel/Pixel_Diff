from __future__ import annotations

import json
from pathlib import Path

from pixel_diff_api.settings import ApiSettings
from pixel_diff_api.task_service import TaskManager


def test_task_manager_runs_comparison_and_reads_result(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "default.yaml"
    config.write_text("dpi: 300", encoding="utf-8")
    settings = ApiSettings(
        project_root=tmp_path,
        task_root=tmp_path / "tasks",
        config_profiles={"default": config},
    )
    task_id = "task-1"
    input_dir = settings.task_root / task_id / "inputs"
    input_dir.mkdir(parents=True)
    template = input_dir / "template.pdf"
    candidate = input_dir / "candidate.pdf"
    template.write_bytes(b"template")
    candidate.write_bytes(b"candidate")

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        report_root = Path(command[command.index("--report-dir") + 1])
        run_dir = report_root / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "diff_result.json").write_text(
            json.dumps({"total_regions": 7, "total_pages": 2}), encoding="utf-8"
        )

        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Result()

    monkeypatch.setattr("pixel_diff_api.task_service.subprocess.run", fake_run)
    manager = TaskManager(settings)
    manager.register(task_id, template, candidate, "default")
    future = manager._futures[task_id]
    future.result(timeout=5)
    task = manager.get(task_id)

    assert task is not None
    assert task.status == "completed"
    assert task.difference_count == 7
    assert task.total_pages == 2
