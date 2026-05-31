from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FeatureConfig, SpikeConfig
from .data_loader import load_csv
from .features import resample_uniform
from .stall_times import (
    STALL_TIMES_PATH,
    _parse_file_entry,
    load_stall_annotations,
    merge_stall_periods,
)

CALIBRATION_REPORT_NAME = "stall_spike_calibration.json"


def discover_calibration_csvs(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []

    def add_from_dir(d: Path) -> None:
        if not d.is_dir():
            return
        for p in sorted(d.glob("*.csv")):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                paths.append(p)

    for root in roots:
        root = Path(root)
        for sub in (
            "normal_motor_tests",
            "stall_motor_tests",
            "motor_data_tests",
            "Data 2/Normal",
            "Data 2/Stalled",
            "data/raw",
        ):
            add_from_dir(root / sub)

    return paths


def _annotation_gaps(stall_times_path: Path | None = None) -> list[float]:
    path = stall_times_path or STALL_TIMES_PATH
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    gaps: list[float] = []
    for value in data.get("files", {}).values():
        parsed = _parse_file_entry(value)
        if not parsed:
            continue
        periods = sorted(parsed.stall_periods_s)
        for i in range(len(periods) - 1):
            gaps.append(periods[i + 1][0] - periods[i][1])
    return gaps


def _auto_stall_segments(
    df: pd.DataFrame,
    min_sustain_s: float = 0.25,
) -> list[tuple[float, float]]:
    if "time_s" not in df.columns:
        df = df.copy()
        df["time_s"] = (df["time_us"] - df["time_us"].iloc[0]) / 1_000_000.0

    mx = float(df["current_a"].max())
    if mx <= 0:
        return []

    thr = max(0.008, 0.25 * mx) if mx < 0.15 else max(0.02, 0.35 * mx)
    above = df["current_a"] >= thr
    times = df["time_s"].values
    segments: list[tuple[float, float]] = []
    i, n = 0, len(df)
    while i < n:
        if not above.iloc[i]:
            i += 1
            continue
        s = i
        while i < n and above.iloc[i]:
            i += 1
        dur = times[i - 1] - times[s]
        if dur >= min_sustain_s:
            segments.append((float(times[s]), float(times[i - 1])))
    return segments


def _inter_segment_gaps(segments: list[tuple[float, float]]) -> list[float]:
    gaps: list[float] = []
    for i in range(len(segments) - 1):
        gaps.append(segments[i + 1][0] - segments[i][1])
    return [g for g in gaps if g >= 0]


def _load_for_analysis(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    df["time_s"] = (df["time_us"] - df["time_us"].iloc[0]) / 1_000_000.0
    return df


def calibrate_spike_thresholds(
    normal_paths: list[Path],
    hz: float = 100.0,
    spike_cfg: SpikeConfig | None = None,
) -> dict:
    spike_cfg = spike_cfg or SpikeConfig()
    window = max(int(spike_cfg.baseline_window_s * hz), 3)

    peak_heights: list[float] = []
    excursion_durations: list[float] = []

    for path in normal_paths:
        df = _load_for_analysis(path)
        rs = resample_uniform(df, hz)
        baseline = rs["current_a"].rolling(window, min_periods=1).median()
        noise = rs["current_a"].rolling(window, min_periods=1).std().fillna(0.0)
        threshold = baseline + spike_cfg.spike_multiplier * noise
        mask = rs["current_a"] > threshold
        peak_heights.extend(rs.loc[mask, "current_a"].tolist())

        m = mask.values
        i, n = 0, len(m)
        while i < n:
            if not m[i]:
                i += 1
                continue
            s = i
            while i < n and m[i]:
                i += 1
            excursion_durations.append((i - s) / hz)

    peaks = np.asarray(peak_heights, dtype=float)
    durs = np.asarray(excursion_durations, dtype=float) if excursion_durations else np.array([0.02])

    min_spike_a = 0.008
    if len(peaks) > 0:
        min_spike_a = float(max(0.008, np.percentile(peaks, 10)))

    min_stall_duration_s = float(max(0.15, np.percentile(durs, 95) * 4.0))
    min_stall_duration_s = float(min(min_stall_duration_s, 0.35))

    return {
        "min_spike_a": round(min_spike_a, 4),
        "min_stall_duration_s": round(min_stall_duration_s, 3),
        "normal_spike_peak_p50_a": round(float(np.median(peaks)), 4) if len(peaks) else None,
        "normal_spike_peak_p99_a": round(float(np.percentile(peaks, 99)), 4) if len(peaks) else None,
        "normal_excursion_duration_p95_s": round(float(np.percentile(durs, 95)), 4),
    }


def calibrate_stall_merge_cooldown(
    stall_paths: list[Path],
    annotation_gaps: list[float],
    stall_times_path: Path | None = None,
) -> dict:
    auto_gaps: list[float] = []
    min_period_durations: list[float] = []

    annotations, _ = load_stall_annotations(stall_times_path)
    for path in stall_paths:
        name = path.name
        if name in annotations:
            for start, end in annotations[name].stall_periods_s:
                min_period_durations.append(end - start)

        df = _load_for_analysis(path)
        segments = _auto_stall_segments(df)
        for s, e in segments:
            min_period_durations.append(e - s)
        auto_gaps.extend(_inter_segment_gaps(segments))

    all_gaps = list(annotation_gaps) + [g for g in auto_gaps if g <= 30.0]
    if not all_gaps:
        recommended = 6.0
    else:
        annotation_max = max(annotation_gaps) if annotation_gaps else 0.0
        short_cluster = [g for g in all_gaps if g <= max(8.0, annotation_max + 1.0)]
        recommended = float(max(annotation_max, np.percentile(short_cluster, 90)))
        recommended = max(2.0, min(recommended + 0.5, 10.0))

    return {
        "stall_merge_cooldown_s": round(recommended, 2),
        "annotation_inter_stall_gaps_s": [round(g, 3) for g in sorted(annotation_gaps)],
        "auto_inter_stall_gap_p90_s": round(float(np.percentile(auto_gaps, 90)), 3)
        if auto_gaps
        else None,
        "min_stall_period_duration_s": round(float(min(min_period_durations)), 3)
        if min_period_durations
        else None,
    }


def run_full_calibration(
    roots: list[Path] | None = None,
    stall_times_path: Path | None = None,
) -> dict:
    roots = roots or [Path(__file__).resolve().parent.parent]
    stall_times_path = stall_times_path or STALL_TIMES_PATH

    all_paths = discover_calibration_csvs(roots)
    normal_paths = [p for p in all_paths if "normal" in p.parent.name.lower()]
    stall_paths = [
        p
        for p in all_paths
        if "stall" in p.parent.name.lower() or "stalled" in p.parent.name.lower()
    ]

    annotation_gaps = _annotation_gaps(stall_times_path)
    spike = calibrate_spike_thresholds(normal_paths)
    merge = calibrate_stall_merge_cooldown(stall_paths, annotation_gaps, stall_times_path)

    annotations, warning_window_s = load_stall_annotations(stall_times_path)
    merged_examples: dict[str, dict] = {}
    cooldown = merge["stall_merge_cooldown_s"]
    if stall_times_path.exists():
        with open(stall_times_path) as f:
            raw_files = json.load(f).get("files", {})
        for name, value in raw_files.items():
            parsed = _parse_file_entry(value)
            if not parsed:
                continue
            raw = parsed.stall_periods_s
            merged = merge_stall_periods(raw, cooldown)
            if merged != raw:
                merged_examples[name] = {
                    "before_s": [[a, b] for a, b in raw],
                    "after_s": [[a, b] for a, b in merged],
                }

    return {
        "roots": [str(r) for r in roots],
        "normal_files": [p.name for p in normal_paths],
        "stall_files": [p.name for p in stall_paths],
        "warning_window_s": warning_window_s,
        **merge,
        **spike,
        "merged_period_examples": merged_examples,
        "notes": {
            "spike_vs_stall": (
                "Spikes: brief current excursions above rolling baseline+mult*std "
                "with duration < min_stall_duration_s (features only). "
                "Stalls: sustained periods from stall_times.json (labels); "
                "consecutive periods within stall_merge_cooldown_s are one chunk."
            ),
        },
    }


def save_calibration_report(
    report: dict,
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or Path(__file__).resolve().parent.parent / "data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / CALIBRATION_REPORT_NAME
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def apply_calibration_to_config(report: dict) -> tuple[float, float, float]:
    return (
        float(report["stall_merge_cooldown_s"]),
        float(report["min_spike_a"]),
        float(report["min_stall_duration_s"]),
    )
