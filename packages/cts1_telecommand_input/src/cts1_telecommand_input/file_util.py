from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parents[4]  # repo root
DEFAULT_DIR = BASE_DIR / "logs" / "telecommands"


def get_default_filename() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M")


def save_command(command: str, filename: str | None) -> Path:
    DEFAULT_DIR.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = get_default_filename()

    # Sanitize filename (avoid weird characters)
    filename = filename.strip().replace(" ", "_")

    filepath = DEFAULT_DIR / f"{filename}.txt"

    timestamp = datetime.now().isoformat()

    with open(filepath, "a", encoding="utf-8") as f:
        # Add separator if file already has content
        if filepath.exists() and filepath.stat().st_size > 0:
            f.write("\n")
        f.write(f"{command}")

    return filepath