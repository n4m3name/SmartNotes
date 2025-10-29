from pathlib import Path
from datetime import datetime

from smartnotes.db import make_engine, ensure_schema
from smartnotes.services.ingest import ingest_file_prepare, upsert_note


def test_ingest_prepare_title_and_wordcount(tmp_path):
    p = tmp_path / "n1.md"
    p.write_text("# Title\nhello world\n", encoding="utf-8")
    r = ingest_file_prepare(p)
    # Allow markdown heading char; normalize for assertion
    assert r.title.lstrip('# ').strip() == "Title"
    assert r.word_count >= 2


def test_upsert_insert_update_skip(run_async, tmp_path):
    # create a temp note file
    p = tmp_path / "n2.md"
    p.write_text("First body", encoding="utf-8")
    r1 = ingest_file_prepare(p)

    # DB setup
    from smartnotes.config import load_settings
    s = load_settings()
    eng, sf = make_engine(s.db_path)

    async def go():
        await ensure_schema(eng)
        async with sf() as session:
            # insert
            st = await upsert_note(session, r1)
            assert st == "inserted"
            await session.commit()

        # update-on-change (same path, different content)
        p.write_text("Changed body", encoding="utf-8")
        r2 = ingest_file_prepare(p)
        async with sf() as session:
            st2 = await upsert_note(session, r2)
            assert st2 == "updated"
            await session.commit()

        # skip (same content hash)
        r3 = ingest_file_prepare(p)
        async with sf() as session:
            st3 = await upsert_note(session, r3)
            assert st3 == "skipped"

    run_async(go())
