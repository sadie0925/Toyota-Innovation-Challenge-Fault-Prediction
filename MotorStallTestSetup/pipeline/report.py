from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import OUTPUT_DIR


def write_results_summary(
    summary: dict,
    metrics: dict,
    output_dir: Path | None = None,
) -> Path:
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "RESULTS_SUMMARY.md"

    test = metrics.get("test", {})
    val = metrics.get("val", {})

    lines = [
        "# Motor Stall Prediction — Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Project goal",
        "",
        "Predict whether a motor is **approaching a stall** and estimate **time until stall**",
        "using only a window of recent current telemetry (predictive maintenance).",
        "",
        "## ML pipeline (challenge steps)",
        "",
        "| Step | What we did | Output |",
        "|------|-------------|--------|",
        "| 1. Problem & data | Stall before failure; `normal_motor_tests/` + `stall_motor_tests/` | Raw CSVs |",
        "| 2. Approach | 2-layer LSTM on current sequence (100 samples @ 100 Hz) | `pipeline/model.py` |",
        "| 3. Clean & prepare | Sanitize, resample, normalize current; labels from stall onset | `data/processed/` |",
        "| 4. Train | Session-based split; BCE risk + SmoothL1 time-to-stall | `data/models/` |",
        "| 5. Results | Metrics, plots, predictions CSV | `data/outputs/` |",
        "",
        "## Model I/O",
        "",
        "- **Input:** last 1.0 s of current (`[100]` values), not raw timestamps",
        "- **Output:** stall risk probability (0–1) + estimated seconds until stall",
        "",
        "## Where to find outputs",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `data/outputs/metrics.json` | Accuracy, precision, recall, F1, ROC-AUC, time-to-stall MAE |",
        "| `data/outputs/training_history.png` | Training / validation loss |",
        "| `data/outputs/dashboard_*.png` | Current, risk, predicted vs actual time-to-stall |",
        "| `data/outputs/predictions_*.csv` | Per-timestep predictions |",
        "| `data/processed/processed_features.csv` | Cleaned data + labels |",
        "| `data/models/stall_lstm.pt` | Trained weights |",
        "",
        "## Test set metrics",
        "",
    ]

    if test:
        lines.extend(
            [
                f"- **Accuracy:** {test.get('accuracy', 0):.3f}",
                f"- **Precision:** {test.get('precision', 0):.3f}",
                f"- **Recall:** {test.get('recall', 0):.3f}",
                f"- **F1:** {test.get('f1', 0):.3f}",
            ]
        )
        if "roc_auc" in test:
            lines.append(f"- **ROC-AUC:** {test['roc_auc']:.3f}")
        if "time_to_stall_mae_s" in test:
            lines.append(
                f"- **Time-to-stall MAE:** {test['time_to_stall_mae_s']:.2f} s "
                f"({test.get('time_to_stall_samples', 0)} pre-stall samples)"
            )
        lines.append(f"- **Actual positive rate:** {test.get('positive_rate_actual', 0):.3f}")
        lines.append(f"- **Predicted positive rate:** {test.get('positive_rate_predicted', 0):.3f}")
    else:
        lines.append("_No test split metrics (add more recordings for held-out sessions)._")

    if val:
        lines.extend(["", "## Validation metrics", ""])
        lines.append(f"- F1: {val.get('f1', 0):.3f} | Precision: {val.get('precision', 0):.3f} | Recall: {val.get('recall', 0):.3f}")

    if "data_files" in summary:
        lines.extend(["", "## Training files", ""])
        for f in summary["data_files"]:
            lines.append(f"- `{f}`")

    path.write_text("\n".join(lines) + "\n")
    return path


def save_metrics_json(metrics: dict, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or OUTPUT_DIR
    path = output_dir / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path
