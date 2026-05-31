"""Discover motor CSV datasets for training and inference."""

from __future__ import annotations

from pathlib import Path

from .config import BASE_DIR, MOTOR_DATA_TESTS_DIR, RAW_DIR


def _sort_key(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if "normal" in name:
        return (0, name)
    if "stall" in name:
        return (2, name)
    return (1, name)


def discover_motor_data_paths(base_dir: Path | None = None) -> list[Path]:
    """
    Return CSV paths from motor_data_tests/ (primary) plus any new files in data/raw/.

    Prefers files matching *normal* or *stall* naming; normal recordings first, stall last.
    """
    base_dir = base_dir or BASE_DIR
    paths: list[Path] = []

    tests_dir = base_dir / "motor_data_tests"
    if tests_dir.exists():
        explicit = {
            *tests_dir.glob("*normal*.csv"),
            *tests_dir.glob("*stall*.csv"),
        }
        if explicit:
            paths.extend(sorted(explicit, key=_sort_key))
        else:
            paths.extend(sorted(tests_dir.glob("*.csv"), key=_sort_key))

    raw_dir = base_dir / "data" / "raw"
    if raw_dir.exists():
        for p in sorted(raw_dir.glob("*.csv")):
            if p not in paths:
                paths.append(p)

    return paths


def pick_stall_file(paths: list[Path]) -> Path:
    for p in paths:
        if "stall" in p.name.lower() and "normal" not in p.name.lower():
            return p
    return paths[-1]
