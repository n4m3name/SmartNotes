from __future__ import annotations
from pathlib import Path
import hashlib
import uuid


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """
    Stream the file and return a hex sha256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def uuid_from_hash(hex_digest: str) -> str:
    """
    Deterministic UUID from any hex digest.
    Using uuid5 with the digest as the 'name' gives a stable ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, hex_digest))


def first_title_line(text: str, limit: int = 80) -> str:
    """
    First non-empty line, stripped; fallback to empty string.
    """
    for line in text.splitlines():
        s = line.strip()
        if s:
            return (s[:limit]).rstrip()
    return ""
