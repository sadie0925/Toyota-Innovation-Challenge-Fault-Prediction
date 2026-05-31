"""Clean and sanitize raw motor current time-series data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SanitizeConfig


def sanitize(df: pd.DataFrame, cfg: SanitizeConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SanitizeConfig()
    out = df.copy()

    out = out.dropna(subset=["time_us", "current_a"])
    out["time_us"] = out["time_us"].astype("int64")
    out["current_a"] = pd.to_numeric(out["current_a"], errors="coerce")
    out = out.dropna(subset=["current_a"])
    out = out.sort_values("time_us").drop_duplicates(subset=["time_us"], keep="first")

    out["current_a"] = out["current_a"].clip(lower=0.0, upper=cfg.max_current_a)

    dt = out["time_us"].diff()
    valid_dt = dt.between(cfg.min_dt_us, cfg.max_dt_us) | dt.isna()
    out = out.loc[valid_dt].reset_index(drop=True)

    if cfg.interpolate_batch_gaps and len(out) > 2:
        out = _interpolate_batch_gaps(out, cfg)

    out["time_s"] = (out["time_us"] - out["time_us"].iloc[0]) / 1_000_000.0
    return out.reset_index(drop=True)


def _interpolate_batch_gaps(df: pd.DataFrame, cfg: SanitizeConfig) -> pd.DataFrame:
    dt = df["time_us"].diff()
    gap_mask = dt > cfg.max_dt_us // 10
    if not gap_mask.any():
        return df

    rows: list[dict] = []
    for i in range(len(df)):
        rows.append(df.iloc[i].to_dict())
        if i < len(df) - 1 and gap_mask.iloc[i + 1]:
            t0, c0 = df.iloc[i]["time_us"], df.iloc[i]["current_a"]
            t1, c1 = df.iloc[i + 1]["time_us"], df.iloc[i + 1]["current_a"]
            step_us = int(np.median(df["time_us"].diff().dropna()))
            if step_us <= 0:
                step_us = 550
            t = t0 + step_us
            while t < t1:
                alpha = (t - t0) / (t1 - t0)
                rows.append(
                    {
                        "time_us": int(t),
                        "current_a": float(c0 + alpha * (c1 - c0)),
                        **({k: df.iloc[i][k] for k in df.columns if k not in ("time_us", "current_a")}),
                    }
                )
                t += step_us
    return pd.DataFrame(rows)
