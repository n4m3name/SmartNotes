from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from smartnotes.utils import sha256_file, uuid_from_hash, first_title_line
from smartnotes.models import Note


def safe_move_to_archive(src: Path, archive_dir: Path) -> Path:
    """
    Move file to archive_dir safely:
    - creates archive_dir
    - avoids overwriting by appending _YYYYmmdd_HHMMSS on collision
    - works across filesystems (uses shutil.move)
    Returns final destination path.
    """
    archive_dir = archive_dir.expanduser()
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / src.name
    if dest.exists():
        stem, ext = src.stem, src.suffix
        ts = time.strftime("%Y%m%d_%H%M%S")
        dest = archive_dir / f"{stem}_{ts}{ext}"
    shutil.move(str(src), str(dest))
    return dest


# -------- discovery --------

def discover_new_notes(root: Path) -> list[Path]:
    """
    Return *.md/*.markdown/*.mdown/*.txt files under the given 'new' root (recursive),
    in a stable sorted order.
    """
    root = root.expanduser()
    if not root.exists() or not root.is_dir():
        return []
    exts = {".md", ".markdown", ".mdown", ".txt"}
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
            files.append(p)
    files.sort()
    return files


# -------- ingest prepare --------

@dataclass
class IngestResult:
    note_id: str
    path: Path
    title: str
    word_count: int
    created_at: datetime
    body_md: str
    inserted: bool = False  # set by upsert_note


def ingest_file_prepare(path: Path) -> IngestResult:
    """
    Read a file, compute deterministic ID, derive title/word_count/timestamps.
    No DB writes here.
    """
    path = path.expanduser()
    # Read as UTF-8; replace invalid bytes so we never crash on odd encodings
    text = path.read_text(encoding="utf-8", errors="replace")

    # Compute content hash → stable UUID
    digest = sha256_file(path)
    note_id = uuid_from_hash(digest)

    # Basic title heuristic
    title = first_title_line(text) or path.stem

    # Word count (simple token split)
    words = len(text.split())

    # Use file mtime for created_at (good default for handwritten notes)
    created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(tzinfo=None)

    return IngestResult(
        note_id=note_id,
        path=path,
        title=title,
        word_count=words,
        created_at=created_at,
        body_md=text,
    )


# -------- DB upsert --------

async def upsert_note(session: AsyncSession, r: IngestResult) -> str:
    """
    Insert or update a note by unique path.
    - If a note with the same path exists and content differs → update fields (id remains stable).
    - If no note exists with that path → insert a new row.
    - If identical content already present → skip.
    Returns one of: "inserted" | "updated" | "skipped".
    """
    # First: if a note with the same content-hash id exists, we consider it unchanged
    existing_by_id = await session.get(Note, r.note_id)
    if existing_by_id is not None:
        r.inserted = False
        return "skipped"

    # Prefer matching by unique path next
    row = (await session.execute(select(Note).where(Note.path == str(r.path)))).scalar_one_or_none()
    if row is not None:
        # Check if content differs (simple body comparison). Title/word_count naturally follow body.
        if (row.body_md or "") == (r.body_md or ""):
            r.inserted = False
            return "skipped"
        # Update fields; keep original id and created_at to preserve identity and history
        row.title = r.title
        row.body_md = r.body_md
        row.word_count = r.word_count
        row.ingested_at = datetime.now(timezone.utc).replace(tzinfo=None)
        r.inserted = False
        return "updated"

    # No existing path: insert new
    session.add(
        Note(
            id=r.note_id,
            path=str(r.path),
            created_at=r.created_at,  # UTC-naive
            ingested_at=datetime.now(timezone.utc).replace(tzinfo=None),
            title=r.title,
            body_md=r.body_md,
            word_count=r.word_count,
        )
    )
    r.inserted = True
    return "inserted"
