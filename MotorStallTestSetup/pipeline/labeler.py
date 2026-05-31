"""Manual stall-time labels (Phase 2): 0 = far from stall, 1 = within warning window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig, LabelConfig
from .stall_times import load_stall_times


def label_stall_events(
    df: pd.DataFrame,
    label_cfg: LabelConfig | None = None,
    feature_cfg: FeatureConfig | None = None,
    spike_cfg=None,
    file_label: str | None = None,
    source_file: str | None = None,
) -> pd.DataFrame:
    """
    Phase 2 labels from manual stall_time (Phase 1, see stall_times.json).

    stall_risk = 1  →  within warning_window_s before stall_time
    stall_risk = 0  →  far from stall, during stall, or after stall
    Normal files  →  all 0

    Spikes are NOT used for labeling — only your annotated stall_time.
    """
    label_cfg = label_cfg or LabelConfig()
    feature_cfg = feature_cfg or FeatureConfig()
    out = df.copy()
    n = len(out)
    times = out["time_s"].values

    stall_risk = np.zeros(n, dtype=np.int32)
    time_to_stall = np.full(n, -1.0, dtype=np.float32)
    stall_phase = np.zeros(n, dtype=np.int32)
    stall_onset_time_s = -1.0

    if file_label == "normal":
        pass
    elif source_file:
        stall_times, warning_window_s = load_stall_times(label_cfg.stall_times_path)
        if label_cfg.warning_window_s is not None:
            warning_window_s = label_cfg.warning_window_s
        stall_time = stall_times.get(source_file)
        if stall_time is not None:
            stall_onset_time_s = float(stall_time)
            warn_start = stall_time - warning_window_s
            for i, t in enumerate(times):
                if t >= stall_time:
                    stall_phase[i] = 1
                elif t >= warn_start:
                    stall_risk[i] = 1
                    time_to_stall[i] = float(stall_time - t)

    out["stall_onset_time_s"] = stall_onset_time_s
    out["stall_risk"] = stall_risk
    out["time_to_stall_s"] = time_to_stall
    out["stall_phase"] = stall_phase
    out["stall_imminent"] = stall_risk
    out["precursor"] = stall_risk
    out["stall_active"] = stall_phase
    out["label_abnormal"] = stall_risk
    return out
