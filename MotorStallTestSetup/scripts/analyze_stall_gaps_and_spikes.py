#!/usr/bin/env python3
"""Analyze stall gaps and spike thresholds from all available motor CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.config import BASE_DIR, DEFAULT_CONFIG
from pipeline.stall_analysis import run_full_calibration, save_calibration_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=[BASE_DIR, BASE_DIR.parent / "MotorStallTestSetup_backup"],
        help="Directories to search for normal/stall CSVs",
    )
    parser.add_argument(
        "--stall-times",
        type=Path,
        default=DEFAULT_CONFIG.label.stall_times_path,
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print suggested PipelineConfig values",
    )
    args = parser.parse_args()

    report = run_full_calibration(args.roots, args.stall_times)
    out_path = save_calibration_report(report)
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")

    if args.print_config:
        print(
            "\nSuggested config updates:\n"
            f"  stall_merge_cooldown_s = {report['stall_merge_cooldown_s']}\n"
            f"  min_spike_a = {report['min_spike_a']}\n"
            f"  min_stall_duration_s = {report['min_stall_duration_s']}\n"
        )


if __name__ == "__main__":
    main()
