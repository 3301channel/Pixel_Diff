from pathlib import Path

from pixel_diff import _app_paths


def test_compare_entry_uses_extensionless_binary_on_frozen_linux(
    tmp_path: Path, monkeypatch
) -> None:
    run_api = tmp_path / "run_api"
    compare = tmp_path / "compare"
    run_api.write_bytes(b"api")
    compare.write_bytes(b"cli")
    monkeypatch.setattr(_app_paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(_app_paths.sys, "executable", str(run_api))
    monkeypatch.setattr(_app_paths.sys, "platform", "linux")

    assert _app_paths.compare_entry() == compare


def test_compare_entry_uses_exe_binary_on_frozen_windows(
    tmp_path: Path, monkeypatch
) -> None:
    run_api = tmp_path / "run_api.exe"
    compare = tmp_path / "compare.exe"
    run_api.write_bytes(b"api")
    compare.write_bytes(b"cli")
    monkeypatch.setattr(_app_paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(_app_paths.sys, "executable", str(run_api))
    monkeypatch.setattr(_app_paths.sys, "platform", "win32")

    assert _app_paths.compare_entry() == compare


def test_compare_entry_prefers_an_existing_sibling_binary(
    tmp_path: Path, monkeypatch
) -> None:
    run_api = tmp_path / "run_api"
    compare_exe = tmp_path / "compare.exe"
    run_api.write_bytes(b"api")
    compare_exe.write_bytes(b"cli")
    monkeypatch.setattr(_app_paths.sys, "frozen", True, raising=False)
    monkeypatch.setattr(_app_paths.sys, "executable", str(run_api))
    monkeypatch.setattr(_app_paths.sys, "platform", "linux")

    assert _app_paths.compare_entry() == compare_exe
