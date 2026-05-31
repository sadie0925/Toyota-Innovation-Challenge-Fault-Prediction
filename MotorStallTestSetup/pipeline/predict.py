"""Inference: stall risk probability + estimated time-to-stall."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import MODEL_DIR, PipelineConfig
from .dataset import build_sequences
from .model import StallLSTM
from .normalize import FeatureNormalizer


def load_trained_model(model_dir: Path | None = None) -> tuple[StallLSTM, FeatureNormalizer, dict]:
    model_dir = model_dir or MODEL_DIR
    with open(model_dir / "model_meta.json") as f:
        meta = json.load(f)

    normalizer = FeatureNormalizer.load(model_dir / "normalizer.joblib")
    model = StallLSTM(
        input_size=len(meta["feature_columns"]),
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
        max_time_to_stall_s=meta.get("max_time_to_stall_s", 5.0),
    )
    model.load_state_dict(torch.load(model_dir / "stall_lstm.pt", map_location="cpu"))
    model.eval()
    return model, normalizer, meta


def predict_dataframe(
    df: pd.DataFrame,
    cfg: PipelineConfig | None = None,
    model_dir: Path | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    cfg = cfg or PipelineConfig()
    model, normalizer, meta = load_trained_model(model_dir)
    if threshold == 0.5 and "risk_threshold" in meta:
        threshold = float(meta["risk_threshold"])

    feature_cols = meta["feature_columns"]
    seq_len = meta["sequence_length"]
    normalized = normalizer.transform(df)

    x, y_risk, y_tts, row_idx = build_sequences(
        normalized, feature_cols, seq_len, risk_col="stall_risk", tts_col="time_to_stall_s"
    )
    if len(x) == 0:
        raise ValueError("Not enough timesteps for prediction.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        risk_logits, tts_pred = model(torch.tensor(x, dtype=torch.float32).to(device))
        probs = torch.sigmoid(risk_logits).cpu().numpy()
        tts = tts_pred.cpu().numpy()

    out = df.loc[row_idx].copy().reset_index(drop=True)
    out["stall_probability"] = probs
    out["stall_predicted"] = (probs >= threshold).astype(int)
    out["time_to_stall_predicted_s"] = tts
    out["time_to_stall_actual_s"] = y_tts
    out["stall_risk_label"] = y_risk.astype(int)

    out["motor_status"] = np.where(
        probs >= 0.8,
        "Critical",
        np.where(probs >= threshold, "Warning", "Normal"),
    )
    return out
