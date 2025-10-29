import os
import asyncio
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def temp_config(tmp_path, monkeypatch):
    notes_dir = tmp_path / "Notes"
    archive_dir = notes_dir / "archive"
    reports_dir = notes_dir / "reports"
    (notes_dir / ".smartnotes").mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cfg = (
        f"notes_dir = \"{notes_dir}\"\n"
        f"archive_dir = \"{archive_dir}\"\n"
        f"reports_dir = \"{reports_dir}\"\n"
        f"vec_model = \"fake-model\"\n"
        f"remote_allowed = false\n"
        f"llm_backend = \"local\"\n"
        f"report_times = {{ daily = \"23:00\", weekly = \"Sun 18:00\", monthly = \"1 18:00\" }}\n"
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(cfg, encoding="utf-8")
    monkeypatch.setenv("SMARTNOTES_CONFIG", str(cfg_path))
    yield
    monkeypatch.delenv("SMARTNOTES_CONFIG", raising=False)


@pytest.fixture
def run_async():
    def _run(coro):
        return asyncio.run(coro)
    return _run
