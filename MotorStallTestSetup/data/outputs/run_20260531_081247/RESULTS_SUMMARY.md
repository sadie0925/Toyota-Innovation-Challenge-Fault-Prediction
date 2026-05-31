# Motor Stall Prediction — Results

Generated: 2026-05-31 08:13

## Project goal

Predict whether a motor is **approaching a stall** and estimate **time until stall**
using only a window of recent current telemetry (predictive maintenance).

## ML pipeline (challenge steps)

| Step | What we did | Output |
|------|-------------|--------|
| 1. Problem & data | Stall before failure; `normal_motor_tests/` + `stall_motor_tests/` | Raw CSVs |
| 2. Approach | 2-layer LSTM on current sequence (100 samples @ 100 Hz) | `pipeline/model.py` |
| 3. Clean & prepare | Sanitize, resample, normalize current; labels from stall onset | `data/processed/` |
| 4. Train | Session-based split; BCE risk + SmoothL1 time-to-stall | `data/models/` |
| 5. Results | Metrics, plots, predictions CSV | `data/outputs/` |

## Model I/O

- **Input:** last 1.0 s of current (`[100]` values), not raw timestamps
- **Output:** stall risk probability (0–1) + estimated seconds until stall

## Where to find outputs

| File | Description |
|------|-------------|
| `data/outputs/metrics.json` | Accuracy, precision, recall, F1, ROC-AUC, time-to-stall MAE |
| `data/outputs/training_history.png` | Training / validation loss |
| `data/outputs/dashboard_*.png` | Current, risk, predicted vs actual time-to-stall |
| `data/outputs/predictions_*.csv` | Per-timestep predictions |
| `data/processed/processed_features.csv` | Cleaned data + labels |
| `data/models/stall_lstm.pt` | Trained weights |

## Test set metrics

- **Accuracy:** 0.642
- **Precision:** 0.524
- **Recall:** 1.000
- **F1:** 0.688
- **ROC-AUC:** 0.775
- **Time-to-stall MAE:** 0.99 s (187 pre-stall samples)
- **Actual positive rate:** 0.394
- **Predicted positive rate:** 0.752

## Validation metrics

- F1: 0.630 | Precision: 0.459 | Recall: 1.000

## Training files

- `/Users/sadiechoi/Desktop/Toyota/S26-Toyota-Innovation-Challenge/Fault_Prediction/MotorStallTestSetup/stall_motor_tests/motor_data_stall.csv`
