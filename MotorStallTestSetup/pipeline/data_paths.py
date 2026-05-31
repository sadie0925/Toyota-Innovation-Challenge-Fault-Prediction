"""Discover motor CSV datasets for training and inference."""

from __future__ import annotations

from pathlib import Path

from .config import BASE_DIR


def discover_motor_data_paths(base_dir: Path | None = None) -> list[Path]:
    """
    Load all CSVs from normal_motor_tests/ then stall_motor_tests/.

    Folder placement defines the label (normal vs stalled); filenames are not required
    to contain "normal" or "stall".
    """
    base_dir = base_dir or BASE_DIR
    paths: list[Path] = []

    normal_dir = base_dir / "normal_motor_tests"
    stall_dir = base_dir / "stall_motor_tests"

    if normal_dir.is_dir():
        paths.extend(sorted(normal_dir.glob("*.csv")))

    if stall_dir.is_dir():
        paths.extend(sorted(stall_dir.glob("*.csv")))

    if paths:
        return paths

    # Legacy flat layout (optional fallback)
    legacy = base_dir / "motor_data_tests"
    if legacy.is_dir():
        paths.extend(sorted(legacy.glob("*.csv")))
        return paths

    raw_dir = base_dir / "data" / "raw"
    if raw_dir.is_dir():
        paths.extend(sorted(raw_dir.glob("*.csv")))

    return paths


def pick_stall_file(paths: list[Path]) -> Path:
    """Pick a stall recording for demo prediction plots."""
    preferred = ("stall", "stalled", "abnormal")
    for p in paths:
        if p.parent.name == "stall_motor_tests" and any(tag in p.name.lower() for tag in preferred):
            return p
    for p in paths:
        name = p.name.lower()
        if "stall" in name and "normal" not in name:
            return p
    return paths[-1]
