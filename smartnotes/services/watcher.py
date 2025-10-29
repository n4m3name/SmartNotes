# smartnotes/services/watcher.py
from __future__ import annotations
from pathlib import Path
import asyncio
import time
import traceback

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from smartnotes.config import load_settings
from smartnotes.db import make_engine
from smartnotes.services.ingest import discover_new_notes, ingest_file_prepare, upsert_note
from smartnotes.services.enrich import enrich_missing
from smartnotes.services.embeddings import build_or_rebuild
from smartnotes.services.ingest import safe_move_to_archive


# ---------------------------------------------------------------------
# Utility: timestamped logging
# ---------------------------------------------------------------------
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------
class _Handler(FileSystemEventHandler):
    """Watches for new/changed notes and triggers ingest → enrich → embed (incrementally)."""

    def __init__(self, loop: asyncio.AbstractEventLoop, notes_new: Path, archive_dir: Path, db_path: Path, vec_model: str):
        super().__init__()
        self._loop = loop
        self.notes_new = notes_new
        self.archive_dir = archive_dir
        self.db_path = db_path
        self.vec_model = vec_model
        self._busy = False

    def on_any_event(self, event):
        """Called from watchdog thread; schedule async work on main loop."""
        if not self._busy:
            self._busy = True
            self._loop.call_soon_threadsafe(self._schedule_run)

    def _schedule_run(self):
        """Debounced execution on main thread."""
        self._loop.call_later(0.75, lambda: asyncio.create_task(self._run()))

    async def _run(self):
        try:
            eng, session_factory = make_engine(self.db_path)
            files = discover_new_notes(self.notes_new)
            if not files:
                self._busy = False
                return

            log(f"Detected {len(files)} new file(s)")

            # INGEST --------------------------------------------------------
            async with session_factory() as session:
                added = 0
                for p in files:
                    try:
                        r = ingest_file_prepare(p)
                        if await upsert_note(session, r):
                            added += 1
                            final = safe_move_to_archive(p, self.archive_dir)
                            log(f"Archived: {final.name}")
                            log(f"Ingested: {p.name}")
                    except Exception as e:
                        log(f"⚠️  Error ingesting {p.name}: {e}")
                        traceback.print_exc()
                if added:
                    await session.commit()
                    log(f"Ingested {added} new note(s).")

                    # ENRICH -------------------------------------------------
                    log("Running enrichment...")
                    await enrich_missing(session)
                    await session.commit()
                    log("Enrichment done.")

            # EMBEDDINGS -----------------------------------------------------
            async with session_factory() as session:
                n_total, dim = await build_or_rebuild(session, self.vec_model, incremental=True)
                await session.commit()
                log(f"Embedding index updated — {n_total} notes total @ dim {dim}.")

        except Exception as e:
            log(f"❌ Watcher run failed: {e}")
            traceback.print_exc()
        finally:
            self._busy = False


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def run_watch():
    """Run SmartNotes watcher (blocking until Ctrl-C)."""
    s = load_settings()
    notes_new = (s.notes_dir / "new").expanduser()
    notes_new.mkdir(parents=True, exist_ok=True)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    handler = _Handler(
        loop=loop,
        notes_new=notes_new,
        archive_dir=s.archive_dir.expanduser(),
        db_path=s.db_path,
        vec_model=s.vec_model,
    )

    observer = Observer()
    observer.schedule(handler, str(notes_new), recursive=True)
    observer.start()

    try:
        log(f"Watching {notes_new} — press Ctrl-C to stop.")
        loop.run_forever()
    except KeyboardInterrupt:
        log("Stopping watcher...")
    finally:
        observer.stop()
        observer.join()
        loop.stop()
        loop.close()
        log("Watcher stopped cleanly.")
