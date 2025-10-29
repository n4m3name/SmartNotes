from pathlib import Path
import tomllib

class Settings:
    def __init__(self, data: dict):
        self.notes_dir   = Path(data.get("notes_dir", "~/Documents/ReflectiveNotes")).expanduser()
        self.archive_dir = Path(data.get("archive_dir", self.notes_dir / "archive")).expanduser()
        self.reports_dir = Path(data.get("reports_dir", self.notes_dir / "reports")).expanduser()
        self.vec_model   = data.get("vec_model", "all-MiniLM-L6-v2")
        self.remote_allowed = bool(data.get("remote_allowed", False))
        self.llm_backend = data.get("llm_backend", "local")
        self.report_times = data.get("report_times", {"daily":"23:00","weekly":"Sun 18:00","monthly":"1 18:00"})

    @property
    def state_dir(self) -> Path:
        return (self.notes_dir / ".smartnotes").expanduser()

    @property
    def db_path(self) -> Path:
        return self.state_dir / "smartnotes.db"

def load_settings(path: str = "config.toml") -> Settings:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    s = Settings(cfg)
    s.state_dir.mkdir(parents=True, exist_ok=True)
    return s
