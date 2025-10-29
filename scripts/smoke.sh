#!/usr/bin/env bash
set -euo pipefail

echo "[smoke] Doctor"
uv run smart doctor || exit 1

NOTE_DIR=$(python - <<'PY'
from smartnotes.config import load_settings
print((load_settings().notes_dir / 'new').expanduser())
PY
)
mkdir -p "$NOTE_DIR"

ts=$(date +%s)
echo "# Smoke One"$'\n'