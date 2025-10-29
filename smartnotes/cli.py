# smartnotes/cli.py
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import click

from smartnotes.config import load_settings
from smartnotes.db import make_engine, ensure_schema
from smartnotes.services.ingest import safe_move_to_archive

from smartnotes.log import setup_logging
setup_logging()


# optional reporters
try:
    from smartnotes.reporters.report import collect, render_md, write_report
except Exception:
    collect = render_md = write_report = None  # reports optional



@click.group()
def cli():
    """SmartNotes CLI — local-first notes agent."""


# ---------------------------------------------------------------------
# init (Step 5): create default config + directories
# ---------------------------------------------------------------------
# Try to use config's canonical path if exported; else default to ~/.config
try:
    from smartnotes.config import DEFAULT_CONFIG_PATH as _CFG_DEFAULT
except Exception:
    _CFG_DEFAULT = "~/.config/smartnotes/config.toml"


@cli.command("init")
@click.option("--force", is_flag=True, help="Overwrite existing config.toml if present.")
def init_command(force: bool) -> None:
    """Create a default config.toml and the notes/archive/reports directories."""
    try:
        import tomli_w  # type: ignore
    except Exception as e:
        raise click.ClickException(
            "tomli_w is required for 'smart init'. Install it with: uv add tomli_w"
        ) from e

    cfg_path = Path(_CFG_DEFAULT).expanduser()
    cfg_dir = cfg_path.parent
    cfg_dir.mkdir(parents=True, exist_ok=True)

    default = {
        "notes_dir": "~/Documents/ReflectiveNotes",
        "archive_dir": "~/Documents/ReflectiveNotes/archive",
        "reports_dir": "~/Documents/ReflectiveNotes/reports",
        "vec_model": "all-MiniLM-L6-v2",
        "remote_allowed": False,
        "llm_backend": "local",  # local | openai | anthropic | gemini
        # Scheduling defaults; you can add an optional "weekly_full" like "Sun 17:30"
        "report_times": {"daily": "23:00", "weekly": "Sun 18:00", "monthly": "1 18:00"},
    }

    if cfg_path.exists() and not force:
        click.echo(f"Config already exists at {cfg_path} (use --force to overwrite).")
    else:
        with open(cfg_path, "wb") as f:
            tomli_w.dump(default, f)
        click.echo(f"Wrote default config to {cfg_path}")

    # Create directories based on the just-written defaults
    notes_dir = Path(default["notes_dir"]).expanduser()
    archive_dir = Path(default["archive_dir"]).expanduser()
    reports_dir = Path(default["reports_dir"]).expanduser()
    for d in (notes_dir, archive_dir, reports_dir, notes_dir / ".smartnotes"):
        d.mkdir(parents=True, exist_ok=True)

    click.echo("Created directories:")
    click.echo(f"  - {notes_dir}")
    click.echo(f"  - {archive_dir}")
    click.echo(f"  - {reports_dir}")
    click.echo(f"  - {notes_dir / '.smartnotes'}")
    click.echo("Done.")


# ---------------------------------------------------------------------
# basics
# ---------------------------------------------------------------------
@cli.command()
def initdb():
    """Create (or update) the database schema and show the DB path."""
    s = load_settings()
    eng, _ = make_engine(s.db_path)

    async def go():
        # Ensure schema via central helper (idempotent)
        await ensure_schema(eng)
        click.echo(f"DB ready: {s.db_path}")

    asyncio.run(go())


@cli.command()
def ping():
    """Trivial DB connectivity check."""
    from sqlalchemy import text
    s = load_settings()
    eng, _ = make_engine(s.db_path)

    async def go():
        await ensure_schema(eng)
        async with eng.connect() as conn:
            out = (await conn.execute(text("SELECT 1"))).scalar_one()
            click.echo(f"DB ok: {out}")

    asyncio.run(go())


# ---------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------
@cli.command()
@click.option("--dry-run", is_flag=True, help="Only list actions; no DB writes or moves.")
@click.option("--archive/--no-archive", default=True, help="Move ingested files to archive/ on success.")
def ingest(dry_run: bool, archive: bool):
    """
    Scan notes_dir/new for new notes, upsert into DB, and optionally move to archive/.
    """
    from smartnotes.services.ingest import (
        discover_new_notes,
        ingest_file_prepare,
        upsert_note,
    )

    s = load_settings()
    new_dir: Path = (s.notes_dir / "new").expanduser()
    arch_dir: Path = (s.archive_dir).expanduser()

    files = discover_new_notes(new_dir)
    if not files:
        click.echo("No new notes.")
        return

    async def go():
        eng, session_factory = make_engine(s.db_path)
        await ensure_schema(eng)
        added = skipped = 0
        async with session_factory() as session:
            for p in files:
                res = ingest_file_prepare(p)
                if dry_run:
                    click.echo(f"WOULD UPSERT: {p.name}  →  {res.note_id}  ({res.word_count} words)")
                    continue
                status = await upsert_note(session, res)
                if status == "inserted":
                    added += 1
                    click.echo(f"ADDED: {p.name}  →  {res.note_id}")
                    if archive:
                        final = safe_move_to_archive(p, arch_dir)
                        click.echo(f"ARCHIVED: {final.name}")
                elif status == "updated":
                    click.echo(f"UPDATED: {p.name}")
                    try:
                        from smartnotes.services.embeddings import mark_dirty
                        mark_dirty()
                    except Exception:
                        pass
                else:
                    skipped += 1
                    click.echo(f"SKIP (no change): {p.name}")
            if not dry_run:
                await session.commit()
        click.echo(f"Done: {added} added, {skipped} skipped.")

    asyncio.run(go())


# ---------------------------------------------------------------------
# enrichment
# ---------------------------------------------------------------------
@cli.command()
@click.option("--ids", multiple=True, help="Specific note IDs; default = all notes missing Meta.")
def enrich(ids: tuple[str, ...]):
    """Rule-based enrichment (mood/sentiment/auto-tags)."""
    from smartnotes.services.enrich import enrich_missing

    s = load_settings()

    async def go():
        eng, session_factory = make_engine(s.db_path)
        await ensure_schema(eng)
        async with session_factory() as session:
            added, skipped = await enrich_missing(session, note_ids=ids or None)
            await session.commit()
            click.echo(f"Enriched: {added} (skipped: {skipped})")

    asyncio.run(go())
@cli.command("enrich-tags")
@click.option("--ids", multiple=True, help="Specific note IDs; default = all notes missing Tags.")
@click.option("--top-k", default=5, show_default=True, help="Max tags to generate per note.")
@click.option("--no-overwrite", is_flag=True, help="Do not overwrite existing tags; skip if present.")
def enrich_tags_cmd(ids: tuple[str, ...], top_k: int, no_overwrite: bool):
    """LLM-based tag enrichment; respects remote_allowed via provider selection."""
    from smartnotes.services.enrich import enrich_tags

    s = load_settings()

    async def go():
        eng, session_factory = make_engine(s.db_path)
        await ensure_schema(eng)
        async with session_factory() as session:
            tagged, skipped = await enrich_tags(session, note_ids=ids or None, overwrite=not no_overwrite, top_k=top_k)
            await session.commit()
            click.echo(f"Tagged: {tagged} (skipped: {skipped})")

    asyncio.run(go())



# ---------------------------------------------------------------------
# embeddings & search
# ---------------------------------------------------------------------
@cli.command()
@click.option("--model", default=None, help="SentenceTransformer model (default: vec_model from config.toml).")
@click.option("--if-dirty", is_flag=True, help="Do a full rebuild only if the dirty flag is set; otherwise incremental.")
@click.option("--full", is_flag=True, help="Force a full rebuild of the index.")
def embed(model: str | None, if_dirty: bool, full: bool):
    """Build or update the embeddings index from all notes."""
    from smartnotes.services.embeddings import build_or_rebuild, is_dirty, clear_dirty
    s = load_settings()
    model_name = model or s.vec_model

    async def go():
        eng, session_factory = make_engine(s.db_path)
        await ensure_schema(eng)
        async with session_factory() as session:
            # Decide rebuild mode
            if full:
                incremental = False
            elif if_dirty:
                incremental = not is_dirty()
            else:
                incremental = True

            n, d = await build_or_rebuild(session, model_name, incremental=incremental)
            await session.commit()
            mode = "incremental" if incremental else "full"
            # Clear dirty after full rebuild
            if not incremental:
                try:
                    clear_dirty()
                except Exception:
                    pass
            click.echo(f"Indexed {n} notes @ dim {d} ({mode}) → vecstore/index.faiss")

    asyncio.run(go())


@cli.command()
@click.argument("q", nargs=-1)
@click.option("-k", default=5, help="Top-K")
@click.option("--model", default=None, help="Model to encode query (default: vec_model).")
def search(q: tuple[str, ...], k: int, model: str | None):
    """Semantic search over notes via FAISS index."""
    from smartnotes.services.embeddings import search_query
    s = load_settings()
    query = " ".join(q).strip()
    if not query:
        click.echo("Provide a query, e.g., smart search attention shadow")
        return
    model_name = model or s.vec_model
    results = search_query(query, k=k, model_name=model_name)
    if not results:
        click.echo("No index yet. Run: smart embed")
        return
    click.echo(f"Query: {query}")
    for nid, score in results:
        click.echo(f"{score:6.3f}  {nid}")


# ---------------------------------------------------------------------
# automation
# ---------------------------------------------------------------------
@cli.command()
def watch():
    """Watch notes_dir/new and auto-ingest → enrich → embed."""
    from smartnotes.services.watcher import run_watch
    run_watch()


@cli.command()
def schedule():
    """Run background scheduler (nightly refresh; weekly report)."""
    from smartnotes.services.scheduler import run_scheduler
    run_scheduler()


@cli.command("rebuild-index")
def rebuild_index() -> None:
    """Force a full rebuild of the embedding index (prunes stale entries)."""
    from smartnotes.services.embeddings import build_or_rebuild

    async def go():
        s = load_settings()
        eng, session_factory = make_engine(s.db_path)
        await ensure_schema(eng)
        async with session_factory() as session:
            n, d = await build_or_rebuild(session, s.vec_model, incremental=False)
            await session.commit()
            click.echo(f"Rebuilt index: {n} notes @ dim {d}")

    asyncio.run(go())


@cli.command("ensure-indexes")
def ensure_indexes() -> None:
    """
    Create recommended indexes if they don't exist (safe for existing SQLite DBs).
    """
    from sqlalchemy import text

    s = load_settings()
    eng, _ = make_engine(s.db_path)

    async def go():
        await ensure_schema(eng)
        stmts = [
            # created_at is common in time-window queries
            "CREATE INDEX IF NOT EXISTS ix_notes_created_at ON notes (created_at)",
            # tag faceting / filtering
            "CREATE INDEX IF NOT EXISTS ix_tags_tag ON tags (tag)",
        ]
        async with eng.begin() as conn:
            for sql in stmts:
                await conn.execute(text(sql))
        click.echo("Indexes ensured: ix_notes_created_at, ix_tags_tag")

    asyncio.run(go())


@cli.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Output JSON summary.")
def doctor_cmd(as_json: bool) -> None:
    """Run diagnostics: config, DB schema, vecstore integrity, and dirty flag."""
    import json as _json
    from smartnotes.doctor import run_checks

    async def go():
        res = await run_checks()
        if as_json:
            click.echo(_json.dumps(res.to_dict(), indent=2))
            return
        # Pretty text output
        click.echo("SmartNotes Doctor\n------------------")
        click.echo(f"OK: {res.ok}")
        if res.errors:
            click.echo("Errors:")
            for e in res.errors:
                click.echo(f"  - {e}")
        if res.warnings:
            click.echo("Warnings:")
            for w in res.warnings:
                click.echo(f"  - {w}")
        # Select a few helpful details
        d = res.details
        click.echo("Details:")
        for k in [
            "config.notes_dir","config.archive_dir","config.reports_dir","config.vec_model",
            "db.tables","db.ping","vecstore.dir","vecstore.exists","vecstore.ntotal","vecstore.ids_count","vecstore.dirty"
        ]:
            if k in d:
                click.echo(f"  {k}: {d[k]}")

    asyncio.run(go())


@cli.command("report")
@click.option(
    "--period",
    type=click.Choice(["daily", "weekly", "monthly"], case_sensitive=False),
    default="daily",
    show_default=True,
    help="Reporting window.",
)
def report_cmd(period: str):
    """
    Generate a Markdown report for the selected period and write it into reports/.
    Requires optional reporters module.
    """
    if not (collect and render_md and write_report):
        raise click.ClickException(
            "Reporting not available (smartnotes.reporters.report not installed)."
        )

    from datetime import datetime, timezone
    s = load_settings()

    async def go():
        eng, session_factory = make_engine(s.db_path)
        await ensure_schema(eng)
        now_utc = datetime.now(timezone.utc)
        async with session_factory() as session:
            bundle = await collect(session, period, now_utc=now_utc)
            md = render_md(bundle)
            path = write_report(md, s.reports_dir, bundle["period"], bundle["start"], bundle["end"])
            await session.commit()
            click.echo(f"Wrote {period} report → {path}")

    asyncio.run(go())


@cli.command("summarize")
@click.option("--id", "note_id", required=True, help="Note ID to summarize.")
@click.option("--sentences", default=5, show_default=True, help="Max sentences in the summary.")
@click.option("--write", "write_out", is_flag=True, help="Write to reports/summaries/<id>.md instead of stdout.")
def summarize_cmd(note_id: str, sentences: int, write_out: bool) -> None:
    """Summarize a note using the configured LLM provider (respects remote_allowed)."""
    from sqlalchemy import select
    from smartnotes.models import Note
    from smartnotes.llm.factory import get_provider

    s = load_settings()
    prov = get_provider()

    async def go():
        eng, sf = make_engine(s.db_path)
        await ensure_schema(eng)
        async with sf() as session:
            row = (await session.execute(select(Note).where(Note.id == note_id))).scalar_one_or_none()
            if row is None:
                raise click.ClickException(f"No note found with id: {note_id}")
            summary = prov.generate_summary(row.body_md or "", max_sentences=sentences)
        if write_out:
            out_dir = (s.reports_dir / "summaries").expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{note_id}.md"
            out.write_text(f"# Summary for {note_id}\n\n{summary}\n", encoding="utf-8")
            click.echo(f"Wrote {out}")
        else:
            click.echo(summary)

    asyncio.run(go())


# allow `uv run -m smartnotes.cli ...` and console entry point
if __name__ == "__main__":
    cli()
