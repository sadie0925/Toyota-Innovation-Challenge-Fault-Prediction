"""Serial data collection from Arduino motor stall test setup."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import serial

from .config import RAW_DIR


def collect_serial(
    port: str,
    baud: int = 115200,
    output_dir: Path | None = None,
    label: str = "normal",
) -> Path:
    """
    Stream time_us,current_A from Arduino and save to CSV.

    Matches the format produced by motorStallTestSetup.ino / saveValues.py.
    """
    output_dir = output_dir or RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"motor_{label}_{timestamp}.csv"

    ser = serial.Serial(port, baud)
    ser.reset_input_buffer()

    print(f"Logging to {out_path} (Ctrl+C to stop)")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_us", "current_A"])
        try:
            while True:
                line = ser.readline().decode(errors="ignore").strip()
                if not line or line == "END":
                    continue
                try:
                    t, current = line.split(",")
                    writer.writerow([int(t), float(current)])
                except ValueError:
                    print(f"Parse error: {line}")
        except KeyboardInterrupt:
            print("\nStopped collection.")
        finally:
            ser.close()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect motor current data over serial")
    parser.add_argument("--port", required=True, help="Serial port (e.g. COM10 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--label", default="normal", choices=["normal", "abnormal", "stalled"])
    parser.add_argument("--output-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()
    path = collect_serial(args.port, args.baud, args.output_dir, args.label)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
