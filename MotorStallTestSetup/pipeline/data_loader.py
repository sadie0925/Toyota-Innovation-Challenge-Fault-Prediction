from __future__ import annotations

from pathlib import Path

import pandas as pd

STANDARD_COLUMNS = ("time_us", "current_a")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    was_ma = False
    for col in df.columns:
        lower = col.lower()
        if lower in ("time_us", "time"):
            rename_map[col] = "time_us"
        elif lower in ("current_a", "current"):
            rename_map[col] = "current_a"
        elif lower in ("current_ma",):
            rename_map[col] = "current_a"
            was_ma = True
        elif lower in ("elapsed_time", "elapsed_time_s", "time_s"):
            rename_map[col] = "time_s"
    df = df.rename(columns=rename_map)

    if "time_us" not in df.columns and "time_s" in df.columns:
        df["time_us"] = (df["time_s"] * 1_000_000).astype("int64")
    if "current_a" in df.columns and was_ma:
        df["current_a"] = df["current_a"] / 1000.0

    missing = [c for c in STANDARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)}")

    return df[list(STANDARD_COLUMNS)].copy()


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    df["source_file"] = path.name
    return df


def load_directory(path: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
    path = Path(path)
    frames = [load_csv(p) for p in sorted(path.glob(pattern))]
    if not frames:
        raise FileNotFoundError(f"No CSV files matching {pattern} in {path}")
    return pd.concat(frames, ignore_index=True)
