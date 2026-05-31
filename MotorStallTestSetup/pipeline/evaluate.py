"""Evaluate trained stall prediction model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import MODEL_DIR, PipelineConfig
from .dataset import SequenceBatch
from .model import StallLSTM


def evaluate_batch(
    batch: SequenceBatch,
    cfg: PipelineConfig | None = None,
    model_dir: Path | None = None,
    threshold: float = 0.5,
) -> dict:
    cfg = cfg or PipelineConfig()
    model_dir = model_dir or MODEL_DIR

    with open(model_dir / "model_meta.json") as f:
        meta = json.load(f)

    model = StallLSTM(
        input_size=len(meta["feature_columns"]),
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(torch.load(model_dir / "stall_lstm.pt", map_location="cpu"))
    model.eval()

    def _predict(x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.array([])
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32))
            return torch.sigmoid(logits).numpy()

    splits = {
        "train": (batch.x_train, batch.y_train),
        "val": (batch.x_val, batch.y_val),
        "test": (batch.x_test, batch.y_test),
    }

    results = {}
    for name, (x, y) in splits.items():
        if len(x) == 0:
            continue
        probs = _predict(x)
        preds = (probs >= threshold).astype(int)
        results[name] = {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, preds).tolist(),
            "classification_report": classification_report(y, preds, zero_division=0),
        }
        if len(np.unique(y)) > 1:
            results[name]["roc_auc"] = float(roc_auc_score(y, probs))

    return results
