"""Load manually annotated stall periods (Phase 1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import BASE_DIR

STALL_TIMES_PATH = BASE_DIR / "stall_times.json"


def merge_stall_periods(
    periods: list[tuple[float, float]],
    cooldown_s: float,
) -> list[tuple[float, float]]:
    """Merge periods when gap (next_start - prev_end) <= cooldown_s."""
    if not periods or cooldown_s <= 0:
        return list(periods)

    ordered = sorted((float(a), float(b)) for a, b in periods)
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        prev_end = merged[-1][1]
        if start - prev_end <= cooldown_s:
            merged[-1][1] = max(prev_end, end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


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


def load_stall_annotations(
    path: Path | None = None,
    merge_cooldown_s: float | None = None,
) -> tuple[dict[str, FileStallAnnotation], float]:
    path = path or STALL_TIMES_PATH
    if not path.exists():
        return {}, 5.0

    with open(path) as f:
        data = json.load(f)

    warning_window_s = float(data.get("warning_window_s", 5.0))
    if merge_cooldown_s is None:
        merge_cooldown_s = float(data.get("stall_merge_cooldown_s", 6.5))

    annotations: dict[str, FileStallAnnotation] = {}
    for name, value in data.get("files", {}).items():
        parsed = _parse_file_entry(value)
        if parsed:
            merged = merge_stall_periods(parsed.stall_periods_s, merge_cooldown_s)
            annotations[name] = FileStallAnnotation(stall_periods_s=merged)
    return annotations, warning_window_s


def load_stall_times(path: Path | None = None) -> tuple[dict[str, float], float]:
    """Legacy helper — first stall period start per file."""
    annotations, warning = load_stall_annotations(path)
    times = {name: ann.stall_periods_s[0][0] for name, ann in annotations.items()}
    return times, warning
