from __future__ import annotations
import asyncio
from datetime import time
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from smartnotes.config import load_settings
from smartnotes.db import make_engine
from smartnotes.services.enrich import enrich_missing
from smartnotes.services.embeddings import build_or_rebuild
from smartnotes.log import logger

def log(msg: str):
    logger.info(msg)



# If you added the report command earlier:
try:
    from smartnotes.reporters.report import collect, render_md, write_report
except Exception:
    collect = render_md = write_report = None  # reports optional


async def job_ingest_enrich_embed():
    s = load_settings()
    eng, session_factory = make_engine(s.db_path)
    async with session_factory() as session:
        # Enrich any notes that don’t have meta yet
        n_enriched, _ = await enrich_missing(session)
        await session.commit()
        # Rebuild embeddings
        await build_or_rebuild(session, s.vec_model)
        await session.commit()
    print("[sched] refresh done")

async def job_weekly_report():
    if not collect:
        print("[sched] report skipped (not enabled)")
        return
    s = load_settings()
    eng, session_factory = make_engine(s.db_path)
    async with session_factory() as session:
        data = await collect(session, period="weekly")
        md = render_md(data)
        path = write_report(md, period="weekly")
        print(f"[sched] wrote {path}")

def run_scheduler():
    s = load_settings()
    tz = ZoneInfo(str(asyncio.get_event_loop().time.__class__.__name__)) if False else None  # we’ll use system tz
    sched = AsyncIOScheduler(timezone=tz)  # None → system/local
    # Nightly refresh (enrich missing + rebuild embeddings)
    sched.add_job(job_ingest_enrich_embed, CronTrigger(hour=23, minute=0))
    # Weekly report Sun 18:00 (change if you want)
    sched.add_job(job_weekly_report, CronTrigger(day_of_week="sun", hour=18, minute=0))
    sched.start()
    print("[sched] started (nightly 23:00; weekly Sun 18:00)")
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass
