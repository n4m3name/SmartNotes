from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import re

import numpy as np
import faiss
from sqlalchemy import inspect, text

from smartnotes.config import load_settings
from smartnotes.db import make_engine, ensure_schema
from smartnotes.services.embeddings import is_dirty


@dataclass
class DoctorResult:
    ok: bool
    warnings: List[str]
    errors: List[str]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _parse_hm(v: str) -> bool:
    return bool(re.match(r"^\d{1,2}:\d{2}$", v.strip()))


def _parse_weekly(v: str) -> bool:
    m = re.match(r"^(mon|tue|wed|thu|fri|sat|sun)\s+\d{1,2}:\d{2}$", v.strip(), re.IGNORECASE)
    return bool(m)


def _parse_monthly(v: str) -> bool:
    m = re.match(r"^\d{1,2}\s+\d{1,2}:\d{2}$", v.strip())
    return bool(m)


async def run_checks() -> DoctorResult:
    s = load_settings()

    warnings: List[str] = []
    errors: List[str] = []
    details: Dict[str, Any] = {}

    # Config and paths
    details["config.notes_dir"] = str(s.notes_dir)
    details["config.archive_dir"] = str(s.archive_dir)
    details["config.reports_dir"] = str(s.reports_dir)
    details["config.vec_model"] = s.vec_model
    details["config.report_times"] = s.report_times

    for p in [s.notes_dir, s.archive_dir, s.reports_dir, s.state_dir]:
        if not Path(p).exists():
            errors.append(f"Missing directory: {p}")

    # Validate report_times formats
    rt = s.report_times or {}
    daily = rt.get("daily")
    weekly = rt.get("weekly")
    monthly = rt.get("monthly")
    weekly_full = rt.get("weekly_full")
    if daily and not _parse_hm(daily):
        warnings.append("report_times.daily format should be 'HH:MM'")
    if weekly and not _parse_weekly(weekly):
        warnings.append("report_times.weekly format should be 'DOW HH:MM' (e.g., 'Sun 18:00')")
    if monthly and not _parse_monthly(monthly):
        warnings.append("report_times.monthly format should be 'D HH:MM' (e.g., '1 18:00')")
    if weekly_full and not _parse_weekly(weekly_full):
        warnings.append("report_times.weekly_full format should be 'DOW HH:MM'")

    # DB and schema
    eng, session_factory = make_engine(s.db_path)
    await ensure_schema(eng)
    async with eng.connect() as conn:
        try:
            val = (await conn.execute(text("SELECT 1"))).scalar_one()
            details["db.ping"] = val
        except Exception as e:
            errors.append(f"DB ping failed: {e}")

    async with eng.begin() as conn:
        def _tables(sync_conn):
            return inspect(sync_conn).get_table_names()
        tables = await conn.run_sync(_tables)
    details["db.tables"] = sorted(tables)
    for t in ("notes", "meta", "tags"):
        if t not in tables:
            errors.append(f"Missing expected table: {t}")

    # Vecstore integrity
    vecdir = (s.notes_dir / ".smartnotes" / "vecstore").expanduser()
    index_path = vecdir / "index.faiss"
    ids_path = vecdir / "note_ids.npy"
    model_txt = vecdir / "model.txt"
    details["vecstore.dir"] = str(vecdir)
    details["vecstore.exists"] = vecdir.exists()

    if index_path.exists() and ids_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            ids = np.load(ids_path)
            details["vecstore.ntotal"] = int(index.ntotal)
            details["vecstore.ids_count"] = int(len(ids))
            if int(index.ntotal) != int(len(ids)):
                warnings.append("Index and ids length mismatch; a full rebuild is recommended.")
        except Exception as e:
            warnings.append(f"Vecstore unreadable: {e}; a full rebuild is recommended.")
    else:
        details["vecstore.status"] = "missing"

    # Model name check
    if model_txt.exists():
        saved_model = model_txt.read_text().strip()
        details["vecstore.model_txt"] = saved_model
        if saved_model != s.vec_model:
            warnings.append("Model in vecstore differs from config.vec_model; searches use vecstore model.")

    # Dirty flag
    details["vecstore.dirty"] = is_dirty()

    ok = (len(errors) == 0)
    return DoctorResult(ok=ok, warnings=warnings, errors=errors, details=details)
