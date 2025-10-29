import os
from pathlib import Path

from smartnotes.db import make_engine, ensure_schema
from smartnotes.services.embeddings import build_or_rebuild, search_query, clear_dirty, mark_dirty
from smartnotes.services.ingest import ingest_file_prepare, upsert_note


def test_embeddings_full_and_incremental(run_async, tmp_path, monkeypatch):
    # Prepare two notes
    n1 = tmp_path / "e1.md"
    n2 = tmp_path / "e2.md"
    n1.write_text("first test note", encoding="utf-8")
    n2.write_text("second test note", encoding="utf-8")

    from smartnotes.config import load_settings
    s = load_settings()
    eng, sf = make_engine(s.db_path)

    # Monkeypatch model loader to avoid network; simple bag-of-words hash to vectors
    import numpy as _np
    class _DummyModel:
        def get_sentence_embedding_dimension(self):
            return 8
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            def _vec(t):
                v = _np.zeros(8, dtype=_np.float32)
                for i, ch in enumerate(t.encode('utf-8')):
                    v[i % 8] += ch
                if normalize_embeddings and _np.linalg.norm(v) > 0:
                    v = v / _np.linalg.norm(v)
                return v
            arr = _np.stack([_vec(t) for t in (texts if isinstance(texts, list) else [texts])])
            return arr
    monkeypatch.setattr('smartnotes.services.embeddings._get_model', lambda name: _DummyModel())

    async def go():
        await ensure_schema(eng)
        async with sf() as session:
            # Insert first note
            st = await upsert_note(session, ingest_file_prepare(n1))
            assert st == "inserted"
            await session.commit()

        # Full build (no existing index)
        async with sf() as session:
            n, d = await build_or_rebuild(session, s.vec_model, incremental=False)
            await session.commit()
            assert n >= 1
            assert d > 0

        # Insert second note, incremental build adds one
        async with sf() as session:
            st2 = await upsert_note(session, ingest_file_prepare(n2))
            assert st2 == "inserted"
            await session.commit()

        async with sf() as session:
            before = n
            n2_total, _ = await build_or_rebuild(session, s.vec_model, incremental=True)
            await session.commit()
            assert n2_total >= before + 1

        # Basic search should return something
        res = search_query("test note", k=2)
        assert len(res) > 0

        # Mark dirty, then do a full rebuild and clear dirty
        mark_dirty()
        async with sf() as session:
            nf, _ = await build_or_rebuild(session, s.vec_model, incremental=False)
            await session.commit()
            assert nf >= n2_total
        clear_dirty()

    run_async(go())
