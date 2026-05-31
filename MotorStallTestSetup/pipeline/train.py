"""Train the LSTM stall predictor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import MODEL_DIR, PipelineConfig
from .dataset import SequenceBatch, make_dataloader
from .model import StallLSTM
from .normalize import FeatureNormalizer


def _pos_weight(y: np.ndarray) -> torch.Tensor:
    n_pos = max(float(y.sum()), 1.0)
    n_neg = max(float(len(y) - n_pos), 1.0)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def train_model(
    batch: SequenceBatch,
    normalizer: FeatureNormalizer,
    cfg: PipelineConfig | None = None,
    model_dir: Path | None = None,
) -> dict:
    cfg = cfg or PipelineConfig()
    model_dir = model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StallLSTM(
        input_size=len(batch.feature_columns),
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    ).to(device)

    train_loader = make_dataloader(
        batch.x_train, batch.y_train, cfg.model.batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        batch.x_val, batch.y_val, cfg.model.batch_size, shuffle=False
    )

    pos_weight = _pos_weight(batch.y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    best_val_loss = float("inf")
    patience = 0
    history: list[dict] = []

    for epoch in range(1, cfg.model.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= max(len(batch.x_train), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
        val_loss /= max(len(batch.x_val), 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), model_dir / "stall_lstm.pt")
        else:
            patience += 1
            if patience >= cfg.model.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    normalizer.save(model_dir / "normalizer.joblib")
    meta = {
        "feature_columns": batch.feature_columns,
        "sequence_length": cfg.model.sequence_length,
        "hidden_size": cfg.model.hidden_size,
        "num_layers": cfg.model.num_layers,
        "dropout": cfg.model.dropout,
        "history": history,
    }
    with open(model_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta
