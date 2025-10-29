from click.testing import CliRunner
import json

from smartnotes.cli import cli
from smartnotes.db import make_engine, ensure_schema
from smartnotes.services.ingest import ingest_file_prepare, upsert_note
from smartnotes.config import load_settings


def test_cli_summarize(monkeypatch, tmp_path, run_async):
    # Insert a note
    p = tmp_path / "c1.md"
    p.write_text("# Heading\nThis is a simple note.", encoding="utf-8")
    s = load_settings()
    eng, sf = make_engine(s.db_path)

    async def go():
        await ensure_schema(eng)
        async with sf() as session:
            r = ingest_file_prepare(p)
            st = await upsert_note(session, r)
            assert st == "inserted"
            await session.commit()
            return r.note_id

    note_id = run_async(go())

    # Run CLI summarize
    runner = CliRunner()
    result = runner.invoke(cli, ["summarize", "--id", note_id, "--sentences", "2"]) 
    assert result.exit_code == 0, result.output
    assert result.output.strip() != ""


def test_cli_doctor_json():
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor", "--json"]) 
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert "details" in data and isinstance(data["details"], dict)
    assert "config.notes_dir" in data["details"]
