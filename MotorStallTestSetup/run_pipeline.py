#!/usr/bin/env python3
"""Run the full motor stall prediction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import BASE_DIR, DEFAULT_CONFIG, MODEL_DIR, OUTPUT_DIR, PipelineConfig
from pipeline.data_paths import discover_motor_data_paths, pick_stall_file
from pipeline.dataset import split_sequences
from pipeline.evaluate import evaluate_batch
from pipeline.normalize import FeatureNormalizer
from pipeline.predict import predict_dataframe
from pipeline.preprocess import preprocess_file, preprocess_paths, save_processed
from pipeline.train import train_model
from pipeline.visualize import plot_current_with_predictions, plot_training_history


def default_data_paths() -> list[Path]:
    paths = discover_motor_data_paths(BASE_DIR)
    if paths:
        return paths
    raise FileNotFoundError(
        "No input CSV files found. Add motor data to motor_data_tests/ or data/raw/."
    )


def run_pipeline(
    data_paths: list[Path] | None = None,
    cfg: PipelineConfig | None = None,
    skip_train: bool = False,
) -> dict:
    cfg = cfg or DEFAULT_CONFIG
    cfg.ensure_dirs()

    paths = data_paths or default_data_paths()
    if not paths:
        raise FileNotFoundError(
            "No input CSV files found. Add motor data to motor_data_tests/ or data/raw/."
        )

    print(f"Processing {len(paths)} file(s)...")
    processed = preprocess_paths(paths, cfg)
    processed_path = save_processed(processed)
    print(f"Saved processed features: {processed_path}")

    normalizer = FeatureNormalizer(cfg.feature.feature_columns)
    normalized = normalizer.fit_transform(processed)

    batch = split_sequences(
        normalized,
        cfg.feature.feature_columns,
        target_col="stall_imminent",
        cfg=cfg.model,
        random_seed=cfg.random_seed,
    )
    print(
        f"Sequences — train: {len(batch.x_train)}, "
        f"val: {len(batch.x_val)}, test: {len(batch.x_test)}"
    )
    print(
        f"Positive labels — train: {batch.y_train.sum():.0f}, "
        f"val: {batch.y_val.sum():.0f}, test: {batch.y_test.sum():.0f}"
    )

    summary: dict = {"processed_path": str(processed_path), "data_files": [str(p) for p in paths]}

    if not skip_train:
        meta = train_model(batch, normalizer, cfg, MODEL_DIR)
        summary["training"] = meta

        history_path = OUTPUT_DIR / "training_history.png"
        plot_training_history(meta["history"], history_path)
        summary["training_plot"] = str(history_path)

        metrics = evaluate_batch(batch, cfg, MODEL_DIR)
        metrics_path = OUTPUT_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        summary["metrics"] = metrics
        summary["metrics_path"] = str(metrics_path)
        print(json.dumps(metrics, indent=2))

    if not skip_train:
        predict_source = pick_stall_file(paths)
        pred_df = predict_dataframe(preprocess_file(predict_source, cfg), cfg, MODEL_DIR)
        pred_out = OUTPUT_DIR / f"predictions_{predict_source.stem}.csv"
        pred_df.to_csv(pred_out, index=False)
        plot_path = OUTPUT_DIR / f"plot_{predict_source.stem}.png"
        plot_current_with_predictions(
            pred_df, plot_path, title=f"Stall Prediction — {predict_source.name}"
        )
        summary["predictions_path"] = str(pred_out)
        summary["plot_path"] = str(plot_path)
        print(f"Predictions saved: {pred_out}")
        print(f"Plot saved: {plot_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor stall LSTM pipeline")
    parser.add_argument(
        "--data",
        nargs="*",
        type=Path,
        help="Input CSV paths (default: all CSVs in motor_data_tests/)",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.epochs:
        cfg.model.epochs = args.epochs

    summary = run_pipeline(args.data, cfg, skip_train=args.skip_train)
    print("\nPipeline complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
