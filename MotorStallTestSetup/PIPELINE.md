# Motor Stall Prediction Pipeline

Predict **will the motor stall within the next 5 seconds?** from recent current telemetry.

## Three phases

### Phase 1 — Annotate stall time (manual)

Edit **`stall_times.json`** and set when each stall recording actually stalls:

```json
{
  "warning_window_s": 5.0,
  "files": {
    "motor_data_stall.csv": 50.0,
    "motor_data_stall2.csv": 112.0
  }
}
```

Watch the CSV plot, note the second the motor stalls, enter that value.

### Phase 2 — Auto labels

| Time range | Label | Meaning |
|------------|-------|---------|
| `t < stall_time - 5s` | **0** | Far from stall |
| `stall_time - 5s ≤ t < stall_time` | **1** | Stall within 5 s |
| `t ≥ stall_time` | **0** (excluded from training) | During / after stall |

Normal files → all **0**. **Spikes are not used for labeling.**

### Phase 3 — LSTM classifier

- **Input:** last 100 current samples (~1 s), not timestamps  
- **Output:** probability of stall within next 5 s  

```bash
python run_pipeline.py
```

## Outputs (never overwritten)

Each run saves to a timestamped folder:

```
data/outputs/run_YYYYMMDD_HHMMSS/
  metrics.json
  RESULTS_SUMMARY.md
  training_history.png
  dashboard_motor_data_stall.png
  predictions_motor_data_stall.csv
  processed_features.csv

data/models/run_YYYYMMDD_HHMMSS/
  stall_lstm.pt
  model_meta.json
  normalizer.joblib
```

## Data layout

```
normal_motor_tests/   ← all labels = 0
stall_motor_tests/    ← labels from stall_times.json
stall_times.json      ← Phase 1 annotations
```

## Why this is better than auto-labeling

The old approach **guessed** stall onset from current spikes / file tail → spikes looked like pre-stall → false alarms. Manual `stall_time` gives ground truth for **when** stall happens; the model learns current patterns **before that moment**, not spike shape alone.
