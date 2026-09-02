from pathlib import Path
from typing import Any


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float while defaulting gracefully on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = ["ensure_directory", "safe_float"]