import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_dependencies_include_pillow() -> None:
    payload = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = payload["project"]["dependencies"]

    assert any(item.lower().startswith("pillow") for item in dependencies)


def test_linux_bundle_explicitly_collects_pillow() -> None:
    spec = (PROJECT_ROOT / "build_linux.spec").read_text(encoding="utf-8")

    assert 'collect_all("PIL")' in spec
