from pathlib import Path
import os
import tomllib

# Canonical default config location (used by `smart init`)
DEFAULT_CONFIG_PATH = "~/.config/smartnotes/config.toml"

def _resolve_config_path(path: str | None) -> Path:
    """
    Resolve the configuration file path in priority order:
    1) Explicit path argument (if provided)
    2) SMARTNOTES_CONFIG environment variable (if set)
    3) ./config.toml in current working directory
    4) ~/.config/smartnotes/config.toml
    Raises FileNotFoundError with guidance if not found.
    """
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env_path = os.getenv("SMARTNOTES_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path("config.toml").absolute())
    candidates.append(Path(DEFAULT_CONFIG_PATH).expanduser())
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No config.toml found. Set SMARTNOTES_CONFIG, place a config.toml in the working directory, "
        "or run 'smart init' to create one at ~/.config/smartnotes/config.toml."
    )

class Settings:
    def __init__(self, data: dict):
        self.notes_dir   = Path(data.get("notes_dir", "~/Documents/ReflectiveNotes")).expanduser()
        self.archive_dir = Path(data.get("archive_dir", self.notes_dir / "archive")).expanduser()
        self.reports_dir = Path(data.get("reports_dir", self.notes_dir / "reports")).expanduser()
        self.vec_model   = data.get("vec_model", "all-MiniLM-L6-v2")
        self.remote_allowed = bool(data.get("remote_allowed", False))
        self.llm_backend = data.get("llm_backend", "local")
        # Scheduling: daily maintenance, weekly report, optional weekly_full rebuild, and monthly placeholder
        self.report_times = data.get(
            "report_times",
            {"daily":"23:00","weekly":"Sun 18:00","monthly":"1 18:00"}
        )

    @property
    def state_dir(self) -> Path:
        return (self.notes_dir / ".smartnotes").expanduser()

    @property
    def db_path(self) -> Path:
        return self.state_dir / "smartnotes.db"

def load_settings(path: str | None = None) -> Settings:
    cfg_path = _resolve_config_path(path)
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    s = Settings(cfg)
    s.state_dir.mkdir(parents=True, exist_ok=True)
    return s
