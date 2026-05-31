# Motor Stall Prediction Pipeline

LSTM-based time-series pipeline for **single-motor stall prediction** using `time_us` and `current_A` telemetry.

## Pipeline Overview

```
Serial / CSV  →  Sanitize  →  Features  →  Label  →  Normalize  →  LSTM  →  Predict
```

| Stage | Module | What it does |
|-------|--------|--------------|
| Collect | `pipeline/collect.py` | Stream data from Arduino over serial |
| Sanitize | `pipeline/sanitize.py` | Drop bad rows, clip current, interpolate UART batch gaps |
| Features | `pipeline/features.py` | Resample to 100 Hz, detect spikes, rolling stats, trend slope |
| Label | `pipeline/labeler.py` | Rule-based stall labels (spike bursts, rising trend) |
| Normalize | `pipeline/normalize.py` | StandardScaler on feature columns |
| Model | `pipeline/model.py` | 2-layer LSTM binary classifier |
| Train | `pipeline/train.py` | Train with BCE loss + early stopping |
| Predict | `pipeline/predict.py` | Output stall probability per timestep |

## Stall Detection Rules

1. **First spike ignored** — represents motor startup inrush, not a fault.
2. **Spike burst** — multiple spikes within a short window → abnormal / stall precursor.
3. **Rising trend** — sustained increase in current slope → stall precursor.
4. **LSTM** — learns temporal patterns from labeled windows to predict stall *before* it happens.

## Quick Start

```bash
cd MotorStallTestSetup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline (uses all CSVs in motor_data_tests/)
python run_pipeline.py
```

Outputs land in `data/processed/`, `data/models/`, and `data/outputs/`.

Training data lives in `motor_data_tests/` — files matching `*normal*.csv` and `*stall*.csv` are picked up automatically.

## Collect New Data

```bash
python -m pipeline.collect --port /dev/ttyUSB0 --label normal
python -m pipeline.collect --port /dev/ttyUSB0 --label stalled
```

CSV format: `time_us,current_A`

## Predict on a Single File

```python
from pipeline.preprocess import preprocess_file
from pipeline.predict import predict_dataframe

df = preprocess_file("motor_data_tests/motor_data_stall.csv")
pred = predict_dataframe(df)
print(pred[["time_s", "current_a", "stall_probability", "stall_predicted"]].tail())
```

## Configuration

Edit thresholds in `pipeline/config.py`:

- `SpikeConfig` — spike sensitivity, startup ignore
- `LabelConfig` — burst count, trend slope, prediction horizon
- `ModelConfig` — sequence length, LSTM size, epochs

## Input Format

Your CSV must have:

```csv
time_us,current_A
2926687,0.000122
2927238,0.000122
```

Compatible aliases (`current_mA`, `elapsed_time`) are auto-converted.
