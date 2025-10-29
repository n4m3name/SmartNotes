# smartnotes/services/embeddings.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

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

    # Load embedding model
    model = SentenceTransformer(model_name)
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
    if index is None:
        # full rebuild
        notes = (await session.execute(select(Note.id, Note.body_md))).all()
        ids = [nid for (nid, _) in notes]
        texts = [body or "" for (_, body) in notes]
        index = faiss.IndexFlatIP(d)
        index.add(model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32))
        np.save(ids_path, np.array(ids))
        model_name_file.write_text(model_name)
        faiss.write_index(index, str(index_path))
        return len(ids), d

    # incremental: find notes missing in index
    rows = (await session.execute(select(Note.id, Note.body_md))).all()
    id_to_body = {nid: body or "" for (nid, body) in rows}
    new_ids = [nid for nid in id_to_body if nid not in existing_ids]
    if not new_ids:
        return len(existing_ids), d

    texts = [id_to_body[nid] for nid in new_ids]
    new_vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    index.add(new_vecs)
    all_ids = existing_ids + new_ids

    # Save
    np.save(ids_path, np.array(all_ids))
    model_name_file.write_text(model_name)
    faiss.write_index(index, str(index_path))
    return len(all_ids), d


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
    model = SentenceTransformer(model_name)
    index = faiss.read_index(str(index_path))
    ids = np.load(ids_path)

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
