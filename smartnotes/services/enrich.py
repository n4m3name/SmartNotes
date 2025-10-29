# smartnotes/services/enrich.py
from __future__ import annotations

from typing import Iterable, Tuple, Optional
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from smartnotes.models import Note, Meta

# Optional VADER sentiment; falls back to a tiny lexicon if unavailable.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _VADER = SentimentIntensityAnalyzer()
except Exception:  # pragma: no cover
    _VADER = None


def _simple_sentiment(text: str) -> Optional[float]:
    """
    Sentiment in [-1, 1]. Prefer VADER; fallback to tiny lexicon if not available.
    Returns None for empty text.
    """
    if not text:
        return None

    # Preferred: VADER compound score
    if _VADER is not None:
        try:
            return float(_VADER.polarity_scores(text).get("compound", 0.0))
        except Exception:
            pass  # fall through to fallback

    # Fallback: very small, deterministic lexicon
    import re
    POS = {"good", "great", "awesome", "love", "happy", "glad", "calm", "progress"}
    NEG = {"bad", "terrible", "hate", "sad", "anxious", "worried", "stuck", "tired"}
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return 0.0
    pos = sum(t in POS for t in tokens)
    neg = sum(t in NEG for t in tokens)
    if pos == neg == 0:
        return 0.0
    score = (pos - neg) / max(1, (pos + neg))
    return max(-1.0, min(1.0, score))


async def enrich_missing(
    session: AsyncSession,
    note_ids: Iterable[str] | None = None,
    *,
    overwrite: bool = True,
) -> Tuple[int, int]:
    """
    Compute sentiment for notes and write a Meta row per note.
    - If `note_ids` is None: process all notes that currently lack a Meta row.
    - If `note_ids` is provided: process exactly those notes.
    - If `overwrite` is True: replace any existing Meta rows for targeted notes.
    Returns: (enriched_count, skipped_count)
    """
    enriched = 0
    skipped = 0

    # Choose target notes
    if note_ids:
        targets = (
            await session.execute(
                select(Note).where(Note.id.in_(list(note_ids)))
            )
        ).scalars().all()
    else:
        # Notes that do not yet have a Meta row
        targets = (
            await session.execute(
                select(Note).where(~Note.id.in_(select(Meta.note_id)))
            )
        ).scalars().all()

    if not targets:
        return 0, 0

    # Optionally clear existing Meta rows for explicit list
    if note_ids and overwrite:
        await session.execute(delete(Meta).where(Meta.note_id.in_(list(note_ids))))

    for n in targets:
        # If not overwriting and meta exists, skip
        if not overwrite:
            existing = (await session.execute(select(Meta).where(Meta.note_id == n.id))).first()
            if existing is not None:
                skipped += 1
                continue

        sent = _simple_sentiment(n.body_md or "")

        # Replace any existing meta for this note (idempotent)
        await session.execute(delete(Meta).where(Meta.note_id == n.id))
        session.add(
            Meta(
                note_id=n.id,
                sentiment=sent,
                mood=None,          # intentionally not set (removed mood words)
                tone=None,          # unused in minimal pipeline
                reading_time_sec=max(1, int(len((n.body_md or "").split()) / 200)) if n.body_md else None,
            )
        )
        enriched += 1

    return enriched, skipped
