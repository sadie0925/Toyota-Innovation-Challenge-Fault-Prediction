"""Discover motor CSV datasets for training and inference."""

from __future__ import annotations

from pathlib import Path

from .config import BASE_DIR


def discover_motor_data_paths(base_dir: Path | None = None) -> list[Path]:
    base_dir = base_dir or BASE_DIR
    paths: list[Path] = []

    normal_dir = base_dir / "normal_motor_tests"
    stall_dir = base_dir / "stall_motor_tests"

    if normal_dir.is_dir():
        paths.extend(sorted(normal_dir.glob("*.csv")))

    if stall_dir.is_dir():
        paths.extend(sorted(stall_dir.glob("*.csv")))

    for sub in ("Data 2/Normal", "Data 2/Stalled"):
        extra = base_dir / sub
        if extra.is_dir():
            paths.extend(sorted(extra.glob("*.csv")))

    backup = base_dir.parent / "MotorStallTestSetup_backup"
    if not paths and backup.is_dir():
        for sub in ("normal_motor_tests", "stall_motor_tests"):
            d = backup / sub
            if d.is_dir():
                paths.extend(sorted(d.glob("*.csv")))

    if paths:
        return paths

    legacy = base_dir / "motor_data_tests"
    if legacy.is_dir():
        paths.extend(sorted(legacy.glob("*.csv")))
        return paths

    raw_dir = base_dir / "data" / "raw"
    if raw_dir.is_dir():
        paths.extend(sorted(raw_dir.glob("*.csv")))

    return paths


def pick_stall_file(paths: list[Path]) -> Path:
    preferred = ("stall", "stalled", "abnormal")
    for p in paths:
        if p.parent.name == "stall_motor_tests" and any(tag in p.name.lower() for tag in preferred):
            return p
    for p in paths:
        name = p.name.lower()
        if "stall" in name and "normal" not in name:
            return p
    return paths[-1]
