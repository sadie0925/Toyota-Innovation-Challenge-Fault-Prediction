"""Run inference with a trained LSTM stall predictor."""

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

    feature_cols = meta["feature_columns"]
    seq_len = meta["sequence_length"]
    normalized = normalizer.transform(df)

    x, _ = build_sequences(normalized, feature_cols, "stall_imminent", seq_len)
    if len(x) == 0:
        raise ValueError("Not enough timesteps for prediction.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

    out = df.iloc[seq_len:].copy().reset_index(drop=True)
    out["stall_probability"] = probs
    out["stall_predicted"] = (probs >= threshold).astype(int)
    return out
