from __future__ import annotations

from pathlib import Path

import pytest

from pixel_diff_api.settings import ApiSettings


def test_api_settings_rejects_unknown_config_and_extension(tmp_path: Path) -> None:
    config = tmp_path / "default.yaml"
    config.write_text("dpi: 300", encoding="utf-8")
    settings = ApiSettings(config_profiles={"default": config})

    assert settings.validate_config_name("default") == config
    assert settings.validate_extension("sample.PDF") == ".pdf"
    with pytest.raises(ValueError, match="unknown config_name"):
        settings.validate_config_name("private-path")
    with pytest.raises(ValueError, match="unsupported file type"):
        settings.validate_extension("malware.exe")
