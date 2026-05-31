#!/usr/bin/env python3
"""Run stall prediction on a single CSV file using a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.config import OUTPUT_DIR
from pipeline.predict import predict_dataframe
from pipeline.preprocess import preprocess_file
from pipeline.visualize import plot_current_with_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict motor stall from a CSV file")
    parser.add_argument("csv", type=Path, help="Input CSV (time_us, current_A)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processed = preprocess_file(args.csv)
    pred = predict_dataframe(processed, threshold=args.threshold)

    out_csv = args.output_dir / f"predictions_{args.csv.stem}.csv"
    out_plot = args.output_dir / f"plot_{args.csv.stem}.png"
    pred.to_csv(out_csv, index=False)
    plot_current_with_predictions(pred, out_plot, title=f"Stall Prediction — {args.csv.name}")

    alerts = pred[pred["stall_predicted"] == 1]
    print(f"Saved predictions: {out_csv}")
    print(f"Saved plot: {out_plot}")
    print(f"Stall alerts: {len(alerts)} / {len(pred)} timesteps")
    if len(alerts):
        print(alerts[["time_s", "current_a", "stall_probability"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
