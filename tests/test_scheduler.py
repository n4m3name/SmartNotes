from pathlib import Path
import numpy as np

from smartnotes.config import load_settings
from smartnotes.services.scheduler import job_ingest_enrich_embed


def test_scheduler_nightly_ingests_and_indexes(run_async, tmp_path, monkeypatch):
    # Monkeypatch model loader to avoid network
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

    s = load_settings()
    new_dir = (s.notes_dir / 'new').expanduser()
    new_dir.mkdir(parents=True, exist_ok=True)
    p = new_dir / 'sched1.md'
    p.write_text('content from scheduler path', encoding='utf-8')

    async def go():
        await job_ingest_enrich_embed()

    run_async(go())

    # Should be archived, not present in new/
    assert not p.exists()

    # Vecstore should exist with at least 1 id
    vecdir = (s.notes_dir / '.smartnotes' / 'vecstore').expanduser()
    ids_path = vecdir / 'note_ids.npy'
    assert ids_path.exists()
    ids = np.load(ids_path)
    assert len(ids) >= 1
