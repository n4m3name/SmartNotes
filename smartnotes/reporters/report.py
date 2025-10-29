# smartnotes/reporters/report.py
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from smartnotes.models import Note, Meta, Tag, Summary
from smartnotes.llm.factory import get_provider


@dataclass
class Window:
    period: str  # daily | weekly | monthly
    start: datetime  # UTC-naive stored as UTC
    end: datetime


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _window_for(period: str, now_utc: datetime | None = None) -> Window:
    period = period.lower()
    now = (now_utc or _utcnow())
    # treat "daily" as last 24h from midnight UTC of today
    today_utc = datetime(now.year, now.month, now.day)
    if period == "daily":
        start = today_utc
        end = start + timedelta(days=1)
    elif period == "weekly":
        # last 7 full days ending today
        end = today_utc + timedelta(days=1)
        start = end - timedelta(days=7)
    elif period == "monthly":
        # last 30 days rolling
        end = today_utc + timedelta(days=1)
        start = end - timedelta(days=30)
    else:
        raise ValueError(f"unknown period: {period}")
    return Window(period=period, start=start, end=end)


async def collect(session: AsyncSession, period: str, now_utc: datetime | None = None) -> dict[str, Any]:
    """
    Collect simple stats for the reporting window:
      - note count
      - average sentiment
      - top tags
      - recent notes (id, title, timestamps)
    """
    win = _window_for(period, now_utc)

    # notes in window
    notes_rows = (
        await session.execute(
            select(Note.id, Note.title, Note.created_at, Note.ingested_at)
            .where(Note.created_at >= win.start, Note.created_at < win.end)
            .order_by(Note.created_at.asc())
        )
    ).all()
    notes = [dict(id=i, title=t or "", created_at=ca, ingested_at=ia) for (i, t, ca, ia) in notes_rows]

    # average sentiment over those notes (join Meta)
    avg_sent = (
        await session.execute(
            select(func.avg(Meta.sentiment))
            .join(Note, Note.id == Meta.note_id)
            .where(Note.created_at >= win.start, Note.created_at < win.end)
        )
    ).scalar()
    avg_sent = float(avg_sent) if avg_sent is not None else None

    # tags frequency over those notes
    tag_rows = (
        await session.execute(
            select(Tag.tag)
            .join(Note, Note.id == Tag.note_id)
            .where(Note.created_at >= win.start, Note.created_at < win.end)
        )
    ).scalars().all()
    counter = Counter(tag_rows)
    top_tags = counter.most_common(10)

    bundle: dict[str, Any] = {
        "period": win.period,
        "start": win.start,
        "end": win.end,
        "note_count": len(notes),
        "avg_sentiment": avg_sent,
        "top_tags": top_tags,
        "notes": notes,
    }
    return bundle


def render_md(bundle: dict[str, Any]) -> str:
    def fmt_dt(dt: datetime) -> str:
        # Render UTC-naive as ISO with Z
        return dt.isoformat(timespec="seconds") + "Z"

    period = bundle["period"]
    start = fmt_dt(bundle["start"])
    end = fmt_dt(bundle["end"])
    n = bundle["note_count"]
    s = bundle["avg_sentiment"]
    tags = bundle["top_tags"]
    notes = bundle["notes"]

    lines: list[str] = []
    lines.append(f"# {period.capitalize()} report ({start} → {end})")
    lines.append("")
    lines.append(f"- Notes: **{n}**")
    if s is not None:
        lines.append(f"- Mean sentiment: **{s:.3f}**")
    else:
        lines.append(f"- Mean sentiment: _n/a_")
    lines.append("")
    lines.append("## Top tags")
    if tags:
        for tag, cnt in tags:
            lines.append(f"- {tag} ({cnt})")
    else:
        lines.append("_(no tags this period)_")
    lines.append("")
    lines.append("## Notes in window")
    if notes:
        for row in notes:
            title = row["title"] or "(untitled)"
            created = fmt_dt(row["created_at"])
            nid = row["id"]
            lines.append(f"- **{title}** — {created}  _(id: {nid})_")
    else:
        lines.append("_(no notes in this window)_")
    lines.append("")
    return "\n".join(lines)


def llm_overview(bundle: dict[str, Any]) -> str:
    """
    Produce a short natural-language overview of the reporting window using the configured provider.
    Respects remote_allowed via provider factory (falls back to local offline summarizer).
    """
    prov = get_provider()
    # Build a minimal text corpus: titles + top tags + counts
    parts: list[str] = []
    parts.append(f"Period: {bundle['period']}")
    parts.append(f"Notes: {bundle['note_count']}")
    if bundle.get("avg_sentiment") is not None:
        parts.append(f"Mean sentiment: {bundle['avg_sentiment']:.3f}")
    tags = ", ".join([t for (t, _) in (bundle.get("top_tags") or [])])
    if tags:
        parts.append(f"Top tags: {tags}")
    titles = "; ".join([(row.get("title") or "(untitled)") for row in (bundle.get("notes") or [])])
    if titles:
        parts.append(f"Titles: {titles}")
    text = "\n".join(parts)
    return prov.generate_summary(text, max_sentences=5)


async def save_period_summary(session: AsyncSession, bundle: dict[str, Any], summary_md: str) -> Summary:
    """
    Persist an LLM-generated overview into the Summary table (period-level cache).
    """
    row = Summary(
        period=bundle["period"],
        start_ts=bundle["start"],
        end_ts=bundle["end"],
        summary_md=summary_md,
        topics_json={k: int(v) for (k, v) in (bundle.get("top_tags") or [])} or None,
        mood_trend_json=None,
        created_at=_utcnow(),
    )
    session.add(row)
    return row


def write_report(markdown: str, reports_dir: Path, period: str, start: datetime, end: datetime) -> Path:
    """
    Write Markdown to reports_dir with a deterministic filename.
    Returns the Path to the written file.
    """
    reports_dir = reports_dir.expanduser()
    reports_dir.mkdir(parents=True, exist_ok=True)

    def ts(d: datetime) -> str:
        # YYYYmmdd for filenames (UTC-naive)
        return d.strftime("%Y%m%d")

    fname = f"{period}_report_{ts(start)}_{ts(end)}.md"
    path = reports_dir / fname
    path.write_text(markdown, encoding="utf-8")
    return path
