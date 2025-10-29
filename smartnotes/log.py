# smartnotes/log.py
from __future__ import annotations
import logging, os

def setup_logging() -> None:
    level_name = os.getenv("SMARTNOTES_LOG", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

logger = logging.getLogger("smartnotes")
