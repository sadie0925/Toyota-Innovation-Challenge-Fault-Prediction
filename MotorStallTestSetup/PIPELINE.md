# Motor Stall Prediction — motor_data_stall.csv

## stall_times.json (your annotations)

```json
"stall_periods_ms": [
  [15000, 17000],
  [23000, 34000],
  [36000, 46000]
]
```

## Auto-generated labels (5 s warning before each period)

| Time | Label | Meaning |
|------|-------|---------|
| 0–10 s | 0 | Far from stall |
| 10–15 s | **1** | Warning before stall #1 |
| 15–17 s | stall | Stall period #1 |
| 18–23 s | **1** | Warning before stall #2 |
| 23–34 s | stall | Stall period #2 |
| 31–36 s | **1** | Warning before stall #3 (31–34 s overlaps stall #2 → excluded) |
| 36–46 s | stall | Stall period #3 |
| 46–55 s | 0 | After last stall |

Counts: **1199** warning samples, **2303** stall-phase samples.

## Run

```bash
python run_pipeline.py          # motor_data_stall.csv only
python run_pipeline.py --epochs 15
```

## Outputs (timestamped, never overwritten)

```
data/outputs/run_YYYYMMDD_HHMMSS/
data/models/run_YYYYMMDD_HHMMSS/
```

Latest successful run: check newest `run_*` folder under `data/outputs/`.
