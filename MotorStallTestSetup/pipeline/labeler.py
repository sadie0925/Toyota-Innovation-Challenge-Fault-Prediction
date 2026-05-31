"""Rule-based stall and precursor labeling for supervised training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig, LabelConfig, SpikeConfig


def label_stall_events(
    df: pd.DataFrame,
    label_cfg: LabelConfig | None = None,
    feature_cfg: FeatureConfig | None = None,
    spike_cfg: SpikeConfig | None = None,
    file_label: str | None = None,
) -> pd.DataFrame:
    """
    Label each timestep with stall_imminent (binary).

    Rules (first spike already ignored in features):
      - Spike burst: >= N spikes within a short window
      - Rising trend: slope above threshold over trend window
      - Sustained high current above adaptive baseline
    """
    label_cfg = label_cfg or LabelConfig()
    feature_cfg = feature_cfg or FeatureConfig()
    spike_cfg = spike_cfg or SpikeConfig()
    hz = feature_cfg.resample_hz

    out = df.copy()
    burst_window = max(int(label_cfg.spike_burst_window_s * hz), 2)
    trend_window = max(int(label_cfg.trend_window_s * hz), 2)
    horizon = max(int(label_cfg.prediction_horizon_s * hz), 1)
    sustain = max(int(label_cfg.stall_sustain_s * hz), 1)

    baseline = out["current_a"].rolling(max(int(spike_cfg.baseline_window_s * hz), 3), min_periods=1).median()
    adaptive_threshold = np.maximum(
        baseline * 2.5,
        label_cfg.stall_current_threshold_a,
    )

    spike_burst = (
        out["spike_flag"].rolling(burst_window, min_periods=1).sum()
        >= label_cfg.spike_burst_count
    )
    rising_trend = out["trend_slope"].rolling(trend_window, min_periods=1).mean() > (
        label_cfg.trend_slope_threshold
    )
    high_current = out["current_a"] >= adaptive_threshold
    sustained_stall = high_current.rolling(sustain, min_periods=1).mean() >= 0.8

    precursor = spike_burst | rising_trend
    out["precursor"] = precursor.astype(int)
    out["stall_active"] = sustained_stall.astype(int)

    if file_label == "normal":
        out["stall_imminent"] = 0
    elif file_label == "stalled":
        out["stall_imminent"] = _label_stalled_recording(out, label_cfg, horizon)
    else:
        out["stall_imminent"] = _shift_precursor_labels(
            precursor | sustained_stall, horizon
        )

    out["label_abnormal"] = (out["stall_imminent"] | out["stall_active"]).astype(int)
    return out


def _label_stalled_recording(df: pd.DataFrame, cfg: LabelConfig, horizon: int) -> pd.Series:
    """
    For known stalled recordings: find when current rises into stall regime,
    then label the preceding horizon as stall_imminent.
    """
    baseline = df["current_a"].rolling(50, min_periods=1).median()
    stall_level = np.maximum(baseline * 3.0, cfg.stall_current_threshold_a)
    onset = df["current_a"] >= stall_level

    if not onset.any():
        return pd.Series(0, index=df.index)

    stall_start = int(onset.idxmax())
    for i in range(len(df)):
        if onset.iloc[i]:
            stall_start = i
            break

    labels = pd.Series(0, index=df.index)
    start = max(0, stall_start - horizon)
    labels.iloc[start:stall_start] = 1
    labels.iloc[stall_start:] = 1
    return labels.astype(int)


def _shift_precursor_labels(signal: pd.Series, horizon: int) -> pd.Series:
    """Mark windows where a stall/precursor event occurs within the next `horizon` steps."""
    arr = signal.astype(int).values
    n = len(arr)
    out = np.zeros(n, dtype=int)
    for i in range(n):
        end = min(n, i + horizon + 1)
        if arr[i:end].max() == 1:
            out[i] = 1
    return pd.Series(out, index=signal.index)
