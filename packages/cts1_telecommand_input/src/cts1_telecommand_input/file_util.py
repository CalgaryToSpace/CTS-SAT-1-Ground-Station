from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

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

    with open(filepath, "a", encoding="utf-8") as f:
        # Add separator if file already has content
        if filepath.exists() and filepath.stat().st_size > 0:
            f.write("\n")
        f.write(f"{command}")

    return filepath

LOCAL_TZ = datetime.now().astimezone().tzinfo

def _get_tz(tz_key: str):
    tz_key = tz_key.upper()

    if tz_key in ("UTC", "UT", "Z"):
        return timezone.utc
    if tz_key in ("LOCAL", "MST"):
        return LOCAL_TZ

    raise ValueError(f"Unsupported timezone: {tz_key}")


def parse_datetime_to_timestamp_ms(dt_str: str) -> int | None:
    if not dt_str:
        return None

    try:
        dt_str = dt_str.strip()

        # FORMAT 1: "YYYY-MM-DD HH:MM TZ"
        if " " in dt_str:
            dt_part, tz_part = dt_str.rsplit(" ", 1)
            dt_part = dt_part.replace(" ", "T")

        # FORMAT 2: "YYYYMMDDHHMM"
        else:
            dt_part = dt_str[:-3]  # YYYYMMDDHHMM → datetime portion
            tz_part = dt_str[-3:]  # last 3 chars = TZ (if provided)

            # If no TZ encoded, default to UTC
            if tz_part.isdigit():
                tz_part = "UTC"
                dt_part = dt_str

            # Convert compact → ISO
            dt_part = datetime.strptime(dt_part[:12], "%Y%m%d%H%M").isoformat()

        tz = _get_tz(tz_part)

        dt = datetime.fromisoformat(dt_part)
        dt = dt.replace(tzinfo=tz)

        return int(dt.timestamp() * 1000)

    except Exception as e:
        return None