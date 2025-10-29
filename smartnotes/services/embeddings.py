# smartnotes/services/embeddings.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict
import asyncio
import os
import tempfile

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from smartnotes.config import load_settings
from smartnotes.models import Note


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def _get_vecstore_dir() -> Path:
    """Return vecstore directory path."""
    s = load_settings()
    d = (s.notes_dir / ".smartnotes" / "vecstore").expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------
# Dirty flag (signals that a full rebuild is needed)
# ---------------------------------------------------------------------
_DIRTY_FLAG = "dirty.flag"

def mark_dirty() -> None:
    d = _get_vecstore_dir()
    (d / _DIRTY_FLAG).write_text("dirty", encoding="utf-8")

def clear_dirty() -> None:
    d = _get_vecstore_dir()
    p = d / _DIRTY_FLAG
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass

def is_dirty() -> bool:
    d = _get_vecstore_dir()
    return (d / _DIRTY_FLAG).exists()


# ---------------------------------------------------------------------
# Rebuild lock (prevents concurrent writers) and atomic writes
# ---------------------------------------------------------------------
_LOCK_FILE = "rebuild.lock"

async def _acquire_lock(timeout: float = 60.0, poll: float = 0.5) -> Path | None:
    d = _get_vecstore_dir()
    lock_path = d / _LOCK_FILE
    start = asyncio.get_event_loop().time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
            return lock_path
        except FileExistsError:
            if (asyncio.get_event_loop().time() - start) >= timeout:
                return None
            await asyncio.sleep(poll)

def _release_lock(lock_path: Path | None) -> None:
    if lock_path and lock_path.exists():
        try:
            lock_path.unlink()
        except Exception:
            pass

def _atomic_save_npy(target: Path, array: np.ndarray) -> None:
    """Atomically write a NumPy array to target by using a temp file handle and replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(target.parent), delete=False) as f:
        tmp_name = f.name
        np.save(f, array)
    os.replace(tmp_name, target)


# ---------------------------------------------------------------------
# Build or update FAISS index
# ---------------------------------------------------------------------
async def build_or_rebuild(session: AsyncSession, model_name: str, *, incremental: bool = True) -> Tuple[int, int]:
    """
    Build or update FAISS index from all notes.
    If incremental=True and an index already exists, only embed new notes.
    Returns (total_notes_indexed, vector_dim).
    """
    vecdir = _get_vecstore_dir()
    index_path = vecdir / "index.faiss"
    ids_path = vecdir / "note_ids.npy"
    model_name_file = vecdir / "model.txt"

    # Load embedding model (cached per-process)
    model = _get_model(model_name)
    d = model.get_sentence_embedding_dimension()

    # Load existing index if present
    index = None
    existing_ids: list[str] = []
    if incremental and index_path.exists() and ids_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            existing_ids = list(np.load(ids_path))
            # ensure model matches; if not, rebuild
            if model_name_file.exists() and model_name_file.read_text().strip() != model_name:
                index = None
                existing_ids = []
        except Exception:
            index = None
            existing_ids = []

    # Select notes to embed
    rows = (await session.execute(select(Note.id, Note.body_md))).all()
    id_to_body = {nid: body or "" for (nid, body) in rows}

    # If no usable index (or model changed), do a full rebuild
    if index is None:
        lock = await _acquire_lock()
        if lock is None:
            # Skip rebuild under contention; caller can retry later
            return 0, d
        try:
            ids = list(id_to_body.keys())
            texts = [id_to_body[nid] for nid in ids]
            index = faiss.IndexFlatIP(d)
            index.add(model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32))
            # Atomic writes via temp files and replace
            _atomic_save_npy(ids_path, np.array(ids))
            tmp_model = model_name_file.with_suffix(".txt.tmp")
            tmp_model.write_text(model_name)
            os.replace(tmp_model, model_name_file)
            tmp_index = index_path.with_suffix(".faiss.tmp")
            faiss.write_index(index, str(tmp_index))
            os.replace(tmp_index, index_path)
            return len(ids), d
        finally:
            _release_lock(lock)

    # Orphan check: if any vectors refer to notes no longer present, fallback to full rebuild
    db_ids = set(id_to_body.keys())
    orphan_ids = [nid for nid in existing_ids if nid not in db_ids]
    if orphan_ids:
        lock = await _acquire_lock()
        if lock is None:
            return len(existing_ids), d
        try:
            ids = list(id_to_body.keys())
            texts = [id_to_body[nid] for nid in ids]
            index = faiss.IndexFlatIP(d)
            index.add(model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32))
            _atomic_save_npy(ids_path, np.array(ids))
            tmp_model = model_name_file.with_suffix(".txt.tmp")
            tmp_model.write_text(model_name)
            os.replace(tmp_model, model_name_file)
            tmp_index = index_path.with_suffix(".faiss.tmp")
            faiss.write_index(index, str(tmp_index))
            os.replace(tmp_index, index_path)
            return len(ids), d
        finally:
            _release_lock(lock)

    # Incremental: find notes missing in index
    new_ids = [nid for nid in id_to_body if nid not in existing_ids]
    if not new_ids:
        return len(existing_ids), d

    texts = [id_to_body[nid] for nid in new_ids]
    new_vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    lock = await _acquire_lock()
    if lock is None:
        return len(existing_ids), d
    try:
        index.add(new_vecs)
        all_ids = existing_ids + new_ids
        _atomic_save_npy(ids_path, np.array(all_ids))
        tmp_model = model_name_file.with_suffix(".txt.tmp")
        tmp_model.write_text(model_name)
        os.replace(tmp_model, model_name_file)
        tmp_index = index_path.with_suffix(".faiss.tmp")
        faiss.write_index(index, str(tmp_index))
        os.replace(tmp_index, index_path)
        return len(all_ids), d
    finally:
        _release_lock(lock)


# ---------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------
def search_query(query: str, *, k: int = 5, model_name: str | None = None) -> List[Tuple[str, float]]:
    """Search FAISS index and return list of (note_id, score)."""
    s = load_settings()
    vecdir = (s.notes_dir / ".smartnotes" / "vecstore").expanduser()
    index_path = vecdir / "index.faiss"
    ids_path = vecdir / "note_ids.npy"
    model_name_file = vecdir / "model.txt"

    if not index_path.exists() or not ids_path.exists():
        return []

    model_name = model_name or (model_name_file.read_text().strip() if model_name_file.exists() else s.vec_model)
    model = _get_model(model_name)
    # Light retry to handle mid-swap files during atomic replace
    for attempt in range(2):
        try:
            index = faiss.read_index(str(index_path))
            ids = np.load(ids_path)
            break
        except Exception:
            if attempt == 1:
                raise
            import time as _t
            _t.sleep(0.2)

    qvec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    scores, idxs = index.search(qvec, k)
    idxs = idxs[0]
    scores = scores[0]

    results = []
    for i, sc in zip(idxs, scores):
        if i < 0 or i >= len(ids):
            continue
        results.append((str(ids[i]), float(sc)))
    return results


# ---------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

def _get_model(model_name: str) -> SentenceTransformer:
    m = _MODEL_CACHE.get(model_name)
    if m is None:
        m = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = m
    return m
