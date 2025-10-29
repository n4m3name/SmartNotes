from __future__ import annotations
import asyncio
import shutil
from pathlib import Path
import click

from smartnotes.config import load_settings
from smartnotes.db import make_engine, ensure_schema
from smartnotes.services.ingest import safe_move_to_archive


@click.group()
def cli():
    """SmartNotes CLI — local-first notes agent."""


# ----- basics -----

@cli.command()
def initdb():
    """Show DB path (migrations create schema)."""
    s = load_settings()
    click.echo(f"DB: {s.db_path}")


@cli.command()
def ping():
    """Trivial DB connectivity check."""
    from sqlalchemy import text
    s = load_settings()
    eng, _ = make_engine(s.db_path)

    async def go():
        async with eng.connect() as conn:
            out = (await conn.execute(text("SELECT 1"))).scalar_one()
            click.echo(f"DB ok: {out}")

    asyncio.run(go())


# ----- ingest -----

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
        added = skipped = 0
        async with session_factory() as session:
            for p in files:
                res = ingest_file_prepare(p)
                if dry_run:
                    click.echo(f"WOULD ADD: {p.name}  →  {res.note_id}  ({res.word_count} words)")
                    continue
                inserted = await upsert_note(session, res)
                if inserted:
                    added += 1
                    click.echo(f"ADDED: {p.name}  →  {res.note_id}")
                    if archive:
                        final = safe_move_to_archive(p, arch_dir)
                        click.echo(f"ARCHIVED: {final.name}")
                else:
                    skipped += 1
                    click.echo(f"SKIP (exists): {p.name}")
            if not dry_run:
                await session.commit()
        click.echo(f"Done: {added} added, {skipped} skipped.")

    asyncio.run(go())


# ----- enrichment -----

@cli.command()
@click.option("--ids", multiple=True, help="Specific note IDs; default = all notes missing Meta.")
def enrich(ids: tuple[str, ...]):
    """Rule-based enrichment (mood/sentiment/auto-tags)."""
    from smartnotes.services.enrich import enrich_missing

    s = load_settings()

    async def go():
        eng, session_factory = make_engine(s.db_path)
        async with session_factory() as session:
            added, skipped = await enrich_missing(session, note_ids=ids or None)
            await session.commit()
            click.echo(f"Enriched: {added} (skipped: {skipped})")

    asyncio.run(go())


# ----- embeddings & search -----

@cli.command()
@click.option("--model", default=None, help="SentenceTransformer model (default: vec_model from config.toml).")
def embed(model: str | None):
    """Rebuild embeddings index from all notes."""
    from smartnotes.db import make_engine
    from smartnotes.services.embeddings import build_or_rebuild
    s = load_settings()
    model_name = model or s.vec_model

    async def go():
        eng, session_factory = make_engine(s.db_path)
        async with session_factory() as session:
            n, d = await build_or_rebuild(session, model_name)
            await session.commit()
            click.echo(f"Indexed {n} notes @ dim {d} → vecstore/index.faiss")

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

# ----- automation -----

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



# allow `uv run -m smartnotes.cli ...`
if __name__ == "__main__":
    cli()
