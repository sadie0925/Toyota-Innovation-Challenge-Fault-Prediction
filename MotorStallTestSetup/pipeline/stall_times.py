"""Load manually annotated stall times (Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path

from .config import BASE_DIR

STALL_TIMES_PATH = BASE_DIR / "stall_times.json"


def load_stall_times(path: Path | None = None) -> tuple[dict[str, float], float]:
    """
    Return ({filename: stall_time_s}, warning_window_s).

    Missing or null entries mean no positive labels for that file until annotated.
    """
    path = path or STALL_TIMES_PATH
    if not path.exists():
        return {}, 5.0

    with open(path) as f:
        data = json.load(f)

    warning_window_s = float(data.get("warning_window_s", 5.0))
    files = data.get("files", {})
    times: dict[str, float] = {}
    for name, value in files.items():
        if value is not None:
            times[name] = float(value)
    return times, warning_window_s


def get_stall_time(filename: str, path: Path | None = None) -> float | None:
    times, _ = load_stall_times(path)
    return times.get(filename)
