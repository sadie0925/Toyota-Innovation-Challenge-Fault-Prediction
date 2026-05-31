"""Feature engineering: resampling, spikes, rolling stats, trend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig, SpikeConfig


def resample_uniform(df: pd.DataFrame, hz: float) -> pd.DataFrame:
    if len(df) < 2:
        return df.copy()

    t_start = df["time_s"].iloc[0]
    t_end = df["time_s"].iloc[-1]
    step = 1.0 / hz
    grid = np.arange(t_start, t_end, step)
    if len(grid) == 0:
        return df.copy()

    # Max aggregation preserves short current spikes that interpolation would wash out.
    bin_idx = np.clip(((df["time_s"].values - t_start) / step).astype(int), 0, len(grid) - 1)
    max_current = np.zeros(len(grid))
    counts = np.zeros(len(grid))
    for i, b in enumerate(bin_idx):
        max_current[b] = max(max_current[b], df["current_a"].iloc[i])
        counts[b] += 1

    empty = counts == 0
    if empty.any():
        filled = np.interp(np.flatnonzero(empty), np.flatnonzero(~empty), max_current[~empty])
        max_current[empty] = filled

    out = pd.DataFrame({"time_s": grid, "current_a": max_current})
    out["time_us"] = (out["time_s"] * 1_000_000).astype("int64")
    if "source_file" in df.columns:
        out["source_file"] = df["source_file"].iloc[0]
    return out


def detect_spikes(
    df: pd.DataFrame,
    spike_cfg: SpikeConfig,
    hz: float,
) -> pd.DataFrame:
    """Mark spikes; first spike in each recording is ignored (motor startup)."""
    out = df.copy()
    window = max(int(spike_cfg.baseline_window_s * hz), 3)
    baseline = out["current_a"].rolling(window, min_periods=1).median()
    noise = out["current_a"].rolling(window, min_periods=1).std().fillna(0.0)
    threshold = np.maximum(
        baseline + spike_cfg.spike_multiplier * noise,
        spike_cfg.min_spike_a,
    )

    raw_spike = (out["current_a"] > threshold).astype(int)
    out["spike_raw"] = raw_spike
    out["startup_spike"] = 0

    first_spike_idx = out.index[raw_spike == 1]
    out["spike_flag"] = raw_spike.copy()
    if len(first_spike_idx) > 0:
        out.loc[first_spike_idx[0], "spike_flag"] = 0
        out.loc[first_spike_idx[0], "startup_spike"] = 1

    return out


def add_features(
    df: pd.DataFrame,
    feature_cfg: FeatureConfig | None = None,
    spike_cfg: SpikeConfig | None = None,
) -> pd.DataFrame:
    feature_cfg = feature_cfg or FeatureConfig()
    spike_cfg = spike_cfg or SpikeConfig()
    hz = feature_cfg.resample_hz

    resampled = resample_uniform(df, hz)
    out = detect_spikes(resampled, spike_cfg, hz)

    roll = max(int(feature_cfg.rolling_window_s * hz), 2)
    out["d_current"] = out["current_a"].diff().fillna(0.0)
    out["rolling_mean"] = out["current_a"].rolling(roll, min_periods=1).mean()
    out["rolling_std"] = out["current_a"].rolling(roll, min_periods=1).std().fillna(0.0)

    spike_window = max(int(spike_cfg.baseline_window_s * hz), 2)
    out["spike_count_window"] = (
        out["spike_flag"].rolling(spike_window, min_periods=1).sum()
    )

    trend_window = max(int(1.0 * hz), 3)
    out["trend_slope"] = _slope_series(out["current_a"], trend_window)

    return out.reset_index(drop=True)


def _slope_series(series: pd.Series, window: int) -> pd.Series:
    """Vectorized rolling linear slope (faster than rolling.apply)."""
    y = series.values.astype(float)
    n = len(y)
    out = np.zeros(n, dtype=float)
    if window < 2:
        return pd.Series(out, index=series.index)

    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)

    for i in range(window - 1, n):
        chunk = y[i - window + 1 : i + 1]
        if denom > 0:
            out[i] = np.sum((x - x_mean) * (chunk - chunk.mean())) / denom
    return pd.Series(out, index=series.index)
