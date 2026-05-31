#!/usr/bin/env python3
"""Run the full motor stall prediction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.config import (
    BASE_DIR,
    DEFAULT_CONFIG,
    MODEL_DIR,
    OUTPUT_DIR,
    PipelineConfig,
    STALL_MOTOR_TESTS_DIR,
    make_run_id,
)
from pipeline.dataset import split_sequences
from pipeline.evaluate import evaluate_batch
from pipeline.normalize import FeatureNormalizer
from pipeline.predict import predict_dataframe
from pipeline.preprocess import preprocess_file, preprocess_paths
from pipeline.report import save_metrics_json, write_results_summary
from pipeline.stall_times import load_stall_annotations
from pipeline.train import train_model
from pipeline.visualize import plot_dashboard, plot_training_history

FOCUS_STALL_FILE = STALL_MOTOR_TESTS_DIR / "motor_data_stall.csv"


def default_data_paths() -> list[Path]:
    if FOCUS_STALL_FILE.exists():
        return [FOCUS_STALL_FILE]
    raise FileNotFoundError(f"Focus file not found: {FOCUS_STALL_FILE}")


def run_pipeline(
    data_paths: list[Path] | None = None,
    cfg: PipelineConfig | None = None,
    skip_train: bool = False,
    run_id: str | None = None,
) -> dict:
    cfg = cfg or DEFAULT_CONFIG
    cfg.ensure_dirs()

    run_id = run_id or make_run_id()
    run_dir = OUTPUT_DIR / f"run_{run_id}"
    model_run_dir = MODEL_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_run_dir.mkdir(parents=True, exist_ok=True)

    paths = data_paths or default_data_paths()
    annotations, warning_window = load_stall_annotations(cfg.label.stall_times_path)

    print(f"Run ID: {run_id}")
    print(f"Output dir: {run_dir}")
    print(f"Model dir:  {model_run_dir}")
    print(f"Warning window: {warning_window}s before each stall period")
    print(f"Processing {len(paths)} file(s)...")
    for p in paths:
        ann = annotations.get(p.name)
        if ann:
            periods = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in ann.stall_periods_s)
            print(f"  - {p.name}  stall periods: [{periods}]")
        else:
            print(f"  - {p.name}  (no annotation)")

    processed = preprocess_paths(paths, cfg)
    processed_path = run_dir / "processed_features.csv"
    processed.to_csv(processed_path, index=False)
    print(f"Saved processed features: {processed_path}")
    print(
        f"Labels — risk={processed['stall_risk'].sum()}, "
        f"stall_phase={processed['stall_phase'].sum()}"
    )

    feature_cols = cfg.feature.feature_columns
    normalizer = FeatureNormalizer(feature_cols)
    normalized = normalizer.fit_transform(processed)

    batch = split_sequences(normalized, feature_cols, cfg=cfg.model, random_seed=cfg.random_seed)
    print(
        f"Sequences — train: {len(batch.x_train)}, "
        f"val: {len(batch.x_val)}, test: {len(batch.x_test)}"
    )
    print(
        f"Within-{warning_window}s labels — train: {batch.y_risk_train.sum():.0f}, "
        f"val: {batch.y_risk_val.sum():.0f}, test: {batch.y_risk_test.sum():.0f}"
    )

    summary: dict = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_dir": str(model_run_dir),
        "labeling": "manual stall periods in stall_times.json",
        "warning_window_s": warning_window,
        "processed_path": str(processed_path),
        "data_files": [str(p) for p in paths],
        "model_input": f"{cfg.model.sequence_length} samples of {feature_cols}",
    }

    if not skip_train:
        meta = train_model(batch, normalizer, cfg, model_run_dir)
        summary["training"] = meta

        plot_training_history(meta["history"], run_dir / "training_history.png")

        metrics = evaluate_batch(batch, cfg, model_run_dir)
        save_metrics_json(metrics, run_dir)
        summary["metrics"] = metrics

        risk_threshold = metrics.get("best_threshold", 0.5)
        with open(model_run_dir / "model_meta.json") as f:
            meta = json.load(f)
        meta["risk_threshold"] = risk_threshold
        meta["run_id"] = run_id
        with open(model_run_dir / "model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        predict_source = paths[0]
        pred_df = predict_dataframe(
            preprocess_file(predict_source, cfg),
            cfg,
            model_run_dir,
            threshold=risk_threshold,
        )

        pred_out = run_dir / f"predictions_{predict_source.stem}.csv"
        pred_df.to_csv(pred_out, index=False)
        dashboard_path = run_dir / f"dashboard_{predict_source.stem}.png"
        plot_dashboard(pred_df, dashboard_path, title=f"Stall Prediction — {predict_source.name}")

        summary["predictions_path"] = str(pred_out)
        summary["dashboard_path"] = str(dashboard_path)

        print(json.dumps(metrics, indent=2))
        print(f"Predictions: {pred_out}")
        print(f"Dashboard: {dashboard_path}")

        results_md = write_results_summary(summary, metrics, run_dir)
        summary["results_summary_path"] = str(results_md)
        print(f"Results summary: {results_md}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor stall LSTM pipeline")
    parser.add_argument(
        "--data",
        nargs="*",
        type=Path,
        help=f"Input CSV (default: {FOCUS_STALL_FILE.name} only)",
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
