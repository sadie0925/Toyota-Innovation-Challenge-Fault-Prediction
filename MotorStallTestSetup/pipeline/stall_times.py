"""Load manually annotated stall periods (Phase 1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import BASE_DIR

STALL_TIMES_PATH = BASE_DIR / "stall_times.json"


@dataclass
class FileStallAnnotation:
    stall_periods_s: list[tuple[float, float]]


def _parse_file_entry(value) -> FileStallAnnotation | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        t = float(value)
        return FileStallAnnotation(stall_periods_s=[(t, t + 2.0)])
    if isinstance(value, dict):
        periods_ms = value.get("stall_periods_ms") or value.get("stall_periods_s")
        if not periods_ms:
            return None
        if "stall_periods_s" in value:
            return FileStallAnnotation(
                stall_periods_s=[(float(a), float(b)) for a, b in periods_ms]
            )
        return FileStallAnnotation(
            stall_periods_s=[(a / 1000.0, b / 1000.0) for a, b in periods_ms]
        )
    return None


def load_stall_annotations(path: Path | None = None) -> tuple[dict[str, FileStallAnnotation], float]:
    path = path or STALL_TIMES_PATH
    if not path.exists():
        return {}, 5.0

    with open(path) as f:
        data = json.load(f)

    warning_window_s = float(data.get("warning_window_s", 5.0))
    annotations: dict[str, FileStallAnnotation] = {}
    for name, value in data.get("files", {}).items():
        parsed = _parse_file_entry(value)
        if parsed:
            annotations[name] = parsed
    return annotations, warning_window_s


def load_stall_times(path: Path | None = None) -> tuple[dict[str, float], float]:
    """Legacy helper — first stall period start per file."""
    annotations, warning = load_stall_annotations(path)
    times = {name: ann.stall_periods_s[0][0] for name, ann in annotations.items()}
    return times, warning
