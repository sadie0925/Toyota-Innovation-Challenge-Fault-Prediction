"""End-to-end preprocessing: sanitize → features → labels → normalize."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, PipelineConfig
from .data_loader import load_csv
from .features import add_features
from .labeler import label_stall_events
from .normalize import FeatureNormalizer
from .sanitize import sanitize


def infer_file_label(path: Path) -> str | None:
    name = path.as_posix().lower()
    if "normal" in name:
        return "normal"
    if any(tag in name for tag in ("stalled", "abnormal", "_stall", "motor_data_stall")):
        return "stalled"
    if name.endswith("stall.csv") or "/stall/" in name:
        return "stalled"
    return None


def preprocess_file(
    path: str | Path,
    cfg: PipelineConfig | None = None,
    file_label: str | None = None,
) -> pd.DataFrame:
    cfg = cfg or PipelineConfig()
    path = Path(path)
    file_label = file_label or infer_file_label(path)

    raw = load_csv(path)
    clean = sanitize(raw, cfg.sanitize)
    featured = add_features(clean, cfg.feature, cfg.spike)
    labeled = label_stall_events(
        featured, cfg.label, cfg.feature, cfg.spike, file_label=file_label
    )
    labeled["source_file"] = path.name
    labeled["file_label"] = file_label or "unknown"
    labeled["session_id"] = 0
    return labeled


def preprocess_paths(
    paths: list[str | Path],
    cfg: PipelineConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or PipelineConfig()
    frames = []
    for session_id, p in enumerate(paths):
        frame = preprocess_file(p, cfg)
        frame["session_id"] = session_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def save_processed(df: pd.DataFrame, name: str = "processed_features.csv") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / name
    df.to_csv(out, index=False)
    return out
