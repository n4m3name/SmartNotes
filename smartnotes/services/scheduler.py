from __future__ import annotations
import asyncio
from datetime import time
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from smartnotes.config import load_settings
from smartnotes.db import make_engine, ensure_schema
from smartnotes.services.ingest import discover_new_notes, ingest_file_prepare, upsert_note, safe_move_to_archive
from smartnotes.services.enrich import enrich_missing
from smartnotes.services.embeddings import build_or_rebuild, mark_dirty, clear_dirty, is_dirty
from smartnotes.log import logger

def log(msg: str):
    logger.info(msg)

# If you added the report command earlier:
try:
    from smartnotes.reporters.report import collect, render_md, write_report
except Exception:
    collect = render_md = write_report = None  # reports optional


async def job_ingest_enrich_embed():
    """Nightly maintenance: ingest any files in new/, enrich, update embeddings."""
    s = load_settings()
    eng, session_factory = make_engine(s.db_path)
    await ensure_schema(eng)

    new_dir = (s.notes_dir / "new").expanduser()
    arch_dir = s.archive_dir.expanduser()
    files = discover_new_notes(new_dir)

    added = 0
    async with session_factory() as session:
        # Ingest
        for p in files:
            r = ingest_file_prepare(p)
            status = await upsert_note(session, r)
            if status == "inserted":
                added += 1
                final = safe_move_to_archive(p, arch_dir)
                logger.info(f"[sched] archived {final.name}")
            elif status == "updated":
                logger.info(f"[sched] updated {p.name}")
                mark_dirty()
            else:
                logger.info(f"[sched] skip (no change): {p.name}")
        if added:
            await session.commit()
            logger.info(f"[sched] ingested {added} note(s)")

        # Enrich (missing)
        n_enriched, _ = await enrich_missing(session)
        if n_enriched:
            await session.commit()
            logger.info(f"[sched] enriched {n_enriched} note(s)")

    # Embeddings (outside same session)
    async with session_factory() as session:
        incremental = not is_dirty()
        n_total, dim = await build_or_rebuild(session, s.vec_model, incremental=incremental)
        await session.commit()
        if not incremental:
            clear_dirty()
        mode = "incremental" if incremental else "full"
        logger.info(f"[sched] embeddings updated ({mode}) — {n_total} notes @ dim {dim}")

async def job_weekly_report():
    if not collect:
        logger.info("[sched] report skipped (not enabled)")
        return
    s = load_settings()
    eng, session_factory = make_engine(s.db_path)
    async with session_factory() as session:
        bundle = await collect(session, period="weekly")
        md = render_md(bundle)
        path = write_report(md, s.reports_dir, bundle["period"], bundle["start"], bundle["end"])
        logger.info(f"[sched] wrote {path}")

async def job_weekly_full_rebuild():
    """Optional safety net: force a full rebuild of the embeddings weekly."""
    s = load_settings()
    eng, session_factory = make_engine(s.db_path)
    async with session_factory() as session:
        n_total, dim = await build_or_rebuild(session, s.vec_model, incremental=False)
        await session.commit()
        clear_dirty()
        logger.info(f"[sched] weekly full rebuild — {n_total} notes @ dim {dim}")

def run_scheduler():
    s = load_settings()
    sched = AsyncIOScheduler()  # use system/local timezone

    # Parse times from config.report_times
    def _parse_hm(hm: str) -> tuple[int, int]:
        h, m = hm.strip().split(":")
        return int(h), int(m)

    # Daily job (nightly maintenance)
    daily = s.report_times.get("daily", "23:00")
    dh, dm = _parse_hm(daily)
    sched.add_job(job_ingest_enrich_embed, CronTrigger(hour=dh, minute=dm))

    # Weekly report (e.g., "Sun 18:00")
    weekly = s.report_times.get("weekly", "Sun 18:00")
    try:
        dow, hm = weekly.split()
        wh, wm = _parse_hm(hm)
        sched.add_job(job_weekly_report, CronTrigger(day_of_week=dow.lower(), hour=wh, minute=wm))
    except Exception:
        # Fallback: Sunday 18:00
        sched.add_job(job_weekly_report, CronTrigger(day_of_week="sun", hour=18, minute=0))

    # Optional: Weekly forced full rebuild of embeddings (e.g., "Sun 17:30")
    weekly_full = s.report_times.get("weekly_full", None)
    if weekly_full:
        try:
            dow, hm = weekly_full.split()
            wh, wm = _parse_hm(hm)
            sched.add_job(job_weekly_full_rebuild, CronTrigger(day_of_week=dow.lower(), hour=wh, minute=wm))
        except Exception:
            # If misconfigured, skip silently but log for visibility
            logger.warning("[sched] invalid report_times.weekly_full format; expected 'DOW HH:MM'")

    # Monthly (e.g., "1 18:00") — optional future job
    # month_spec = s.report_times.get("monthly", "1 18:00")
    # try:
    #     dom, hm = month_spec.split()
    #     mh, mm = _parse_hm(hm)
    #     sched.add_job(job_monthly_report, CronTrigger(day=int(dom), hour=mh, minute=mm))
    # except Exception:
    #     pass

    sched.start()
    logger.info("[sched] started using config.report_times")
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass
