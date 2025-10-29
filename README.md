# SmartNotes

Local-first notes pipeline: ingest → enrich → embed → search/report.

## Installation

- Ensure Python and uv are installed.
- Install dependencies and console entry:

```
uv sync
```

## Quickstart

- Initialize configuration and directories:

```
uv run smart init
uv run smart initdb
```

- Put Markdown files into `~/Documents/ReflectiveNotes/new/`.

- Ingest and archive:

```
uv run smart ingest
```

- Enrich metadata (mood/sentiment/tags):

```
uv run smart enrich
```

## Embeddings

Build or update the FAISS index:

```
uv run smart embed
```

Embedding rebuild policy:

- Incremental by default: only new notes are added to the index.
- Dirty flag: when an existing note is updated, we mark a small flag file at `<notes_dir>/.smartnotes/vecstore/dirty.flag`.
- Full rebuild is performed when the dirty flag is set, ensuring updated notes are re-embedded and stale vectors are pruned.

CLI flags:

- `uv run smart embed --if-dirty` — do a full rebuild only if the dirty flag is set; otherwise incremental.
- `uv run smart embed --full` — force a full rebuild regardless of the dirty flag.

## Search

```
uv run smart search your query here
```

## Scheduler

Start the background scheduler with:

```
uv run smart schedule
```

This runs nightly maintenance (ingest → enrich → embeddings) at the time in `report_times.daily` and a weekly report at `report_times.weekly`.

Optional: add a weekly forced full rebuild of embeddings by setting `report_times.weekly_full` to a value like "Sun 17:30". If set, this job will rebuild embeddings in full and clear the dirty flag as a safety net.

## Reports

Generate ad-hoc reports:

```
uv run smart report --period weekly
```

Scheduler will also produce weekly reports into your configured `reports_dir`.

## Configuration

The config file is resolved in this order:
1. Explicit path (CLI or API)
2. SMARTNOTES_CONFIG env var
3. ./config.toml in the current directory
4. ~/.config/smartnotes/config.toml (canonical default)

Default keys:

```toml
notes_dir = "~/Documents/ReflectiveNotes"
archive_dir = "~/Documents/ReflectiveNotes/archive"
reports_dir = "~/Documents/ReflectiveNotes/reports"
vec_model = "all-MiniLM-L6-v2"
remote_allowed = false
llm_backend = "local" # local | openai | anthropic | gemini

[report_times]
daily = "23:00"
weekly = "Sun 18:00"
monthly = "1 18:00"
# optional: weekly forced full embeddings rebuild
# weekly_full = "Sun 17:30"
```

## Notes

- Vector store lives at `<notes_dir>/.smartnotes/vecstore`.
- SQLite DB lives at `<notes_dir>/.smartnotes/smartnotes.db`.
- Use `uv run smart rebuild-index` to force a full rebuild and prune stale vectors.
