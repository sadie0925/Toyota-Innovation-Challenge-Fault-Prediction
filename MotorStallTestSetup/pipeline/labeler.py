"""Manual stall-period labels (Phase 2): 0 = far, 1 = within warning window before stall."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig, LabelConfig
from .stall_times import load_stall_annotations


def label_stall_events(
    df: pd.DataFrame,
    label_cfg: LabelConfig | None = None,
    feature_cfg: FeatureConfig | None = None,
    spike_cfg=None,
    file_label: str | None = None,
    source_file: str | None = None,
) -> pd.DataFrame:

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
        annotations, warning_window_s = load_stall_annotations(
            label_cfg.stall_times_path,
            merge_cooldown_s=label_cfg.stall_merge_cooldown_s,
        )
        if label_cfg.warning_window_s is not None:
            warning_window_s = label_cfg.warning_window_s

        ann = annotations.get(source_file)
        if ann:
            periods = ann.stall_periods_s
            stall_onset_time_s = float(periods[0][0])
            # One warning window before the first chunk only (merged re-stalls)

            for i, t in enumerate(times):
                in_stall = any(start <= t <= end for start, end in periods)
                if in_stall:
                    stall_phase[i] = 1
                    continue

                for start, _end in periods:
                    warn_start = start - warning_window_s
                    if warn_start <= t < start:
                        stall_risk[i] = 1
                        time_to_stall[i] = float(start - t)
                        break

    out["stall_onset_time_s"] = stall_onset_time_s
    out["stall_risk"] = stall_risk
    out["time_to_stall_s"] = time_to_stall
    out["stall_phase"] = stall_phase
    out["stall_imminent"] = stall_risk
    out["precursor"] = stall_risk
    out["stall_active"] = stall_phase
    out["label_abnormal"] = stall_risk
    return out
