"""Evaluate stall risk + time-to-stall on held-out sessions."""

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
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import MODEL_DIR, PipelineConfig
from .dataset import SequenceBatch
from .model import StallLSTM


def find_best_threshold(probs: np.ndarray, y: np.ndarray) -> float:
    """Pick threshold that maximizes F1 on validation data."""
    if len(y) == 0 or y.sum() == 0:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 17):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


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
        max_time_to_stall_s=meta.get("max_time_to_stall_s", 5.0),
    )
    model.load_state_dict(torch.load(model_dir / "stall_lstm.pt", map_location="cpu"))
    model.eval()

    def _predict(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(x) == 0:
            return np.array([]), np.array([])
        with torch.no_grad():
            risk_logits, tts = model(torch.tensor(x, dtype=torch.float32))
            return torch.sigmoid(risk_logits).numpy(), tts.numpy()

    splits = {
        "train": (batch.x_train, batch.y_risk_train, batch.y_tts_train),
        "val": (batch.x_val, batch.y_risk_val, batch.y_tts_val),
        "test": (batch.x_test, batch.y_risk_test, batch.y_tts_test),
    }

    results = {}
    best_threshold = 0.5
    if len(batch.x_val) and batch.y_risk_val.sum() > 0:
        val_probs, _ = _predict(batch.x_val)
        best_threshold = find_best_threshold(val_probs, batch.y_risk_val)
    results["best_threshold"] = best_threshold
    threshold = best_threshold

    for name, (x, y_risk, y_tts) in splits.items():
        if len(x) == 0:
            continue
        probs, tts_pred = _predict(x)
        preds = (probs >= threshold).astype(int)
        block: dict = {
            "threshold": threshold,
            "accuracy": float(accuracy_score(y_risk, preds)),
            "precision": float(precision_score(y_risk, preds, zero_division=0)),
            "recall": float(recall_score(y_risk, preds, zero_division=0)),
            "f1": float(f1_score(y_risk, preds, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_risk, preds).tolist(),
            "classification_report": classification_report(y_risk, preds, zero_division=0),
            "positive_rate_actual": float(y_risk.mean()),
            "positive_rate_predicted": float(preds.mean()),
        }
        if len(np.unique(y_risk)) > 1:
            block["roc_auc"] = float(roc_auc_score(y_risk, probs))

        mask = y_risk > 0.5
        if mask.sum() > 0:
            block["time_to_stall_mae_s"] = float(mean_absolute_error(y_tts[mask], tts_pred[mask]))
            block["time_to_stall_samples"] = int(mask.sum())

        results[name] = block

    return results
