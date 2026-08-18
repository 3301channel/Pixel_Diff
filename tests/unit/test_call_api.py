from pathlib import Path

import pytest

from scripts.call_api import ApiClientError, build_multipart_body, main, parse_args, wait_for_task


def test_build_multipart_body_contains_fields_filenames_and_file_bytes(tmp_path: Path) -> None:
    template = tmp_path / "template.pdf"
    candidate = tmp_path / "candidate.docx"
    template.write_bytes(b"%PDF-template")
    candidate.write_bytes(b"PK-candidate")

    body, content_type = build_multipart_body(
        {"config_name": "sensitive_recall_trial"},
        {
            "template_file": template,
            "candidate_file": candidate,
        },
    )

    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="config_name"' in body
    assert b"sensitive_recall_trial" in body
    assert b'name="template_file"; filename="template.pdf"' in body
    assert b'name="candidate_file"; filename="candidate.docx"' in body
    assert b"%PDF-template" in body
    assert b"PK-candidate" in body


def test_wait_for_task_returns_completed_payload() -> None:
    responses = iter(
        [
            {"task_id": "abc", "status": "pending"},
            {"task_id": "abc", "status": "running"},
            {
                "task_id": "abc",
                "status": "completed",
                "total_pages": 2,
                "difference_count": 7,
            },
        ]
    )

    result = wait_for_task(
        "http://127.0.0.1:8000",
        "abc",
        timeout=10,
        poll_interval=0,
        request_json_fn=lambda _url: next(responses),
        sleep_fn=lambda _seconds: None,
    )

    assert result["status"] == "completed"
    assert result["total_pages"] == 2
    assert result["difference_count"] == 7


def test_wait_for_task_raises_service_error_for_failed_task() -> None:
    with pytest.raises(ApiClientError, match="comparison failed: alignment failed"):
        wait_for_task(
            "http://127.0.0.1:8000",
            "abc",
            timeout=10,
            poll_interval=0,
            request_json_fn=lambda _url: {
                "task_id": "abc",
                "status": "failed",
                "error": "alignment failed",
            },
            sleep_fn=lambda _seconds: None,
        )


def test_wait_for_task_times_out_while_pending() -> None:
    clock = iter([10.0, 10.0, 11.1])

    with pytest.raises(ApiClientError, match="timed out after 1.0 seconds"):
        wait_for_task(
            "http://127.0.0.1:8000",
            "abc",
            timeout=1,
            poll_interval=0,
            request_json_fn=lambda _url: {"task_id": "abc", "status": "pending"},
            sleep_fn=lambda _seconds: None,
            monotonic_fn=lambda: next(clock),
        )


def test_parse_args_uses_local_api_and_sensitive_profile_by_default(tmp_path: Path) -> None:
    template = tmp_path / "template.pdf"
    candidate = tmp_path / "candidate.pdf"

    args = parse_args([str(template), str(candidate)])

    assert args.base_url == "http://127.0.0.1:8000"
    assert args.config_name == "sensitive_recall_trial"
    assert args.poll_interval == 1.0
    assert args.timeout == 900.0
    assert args.download_reports is False
    assert args.no_images is False


def test_main_rejects_missing_input_before_sending_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing_template = tmp_path / "missing.pdf"
    candidate = tmp_path / "candidate.pdf"
    candidate.write_bytes(b"candidate")

    exit_code = main([str(missing_template), str(candidate)])

    assert exit_code == 2
    assert f"input file does not exist: {missing_template}" in capsys.readouterr().err
