"""Train dual-head LSTM (stall risk + time-to-stall)."""

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
    max_tts = cfg.label.warning_window_s

    model = StallLSTM(
        input_size=len(batch.feature_columns),
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        max_time_to_stall_s=max_tts,
    ).to(device)

    train_loader = make_dataloader(
        batch.x_train, batch.y_risk_train, batch.y_tts_train, cfg.model.batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        batch.x_val, batch.y_risk_val, batch.y_tts_val, cfg.model.batch_size, shuffle=False
    )

    pos_weight = _pos_weight(batch.y_risk_train).to(device)
    risk_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    tts_criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)

    best_val_loss = float("inf")
    patience = 0
    history: list[dict] = []

    for epoch in range(1, cfg.model.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, y_risk, y_tts in train_loader:
            xb = xb.to(device)
            y_risk = y_risk.to(device)
            y_tts = y_tts.to(device)
            optimizer.zero_grad()
            risk_logits, tts_pred = model(xb)
            loss_risk = risk_criterion(risk_logits, y_risk)
            mask = y_risk > 0.5
            if mask.any():
                loss_tts = tts_criterion(tts_pred[mask], y_tts[mask]).mean()
            else:
                loss_tts = torch.tensor(0.0, device=device)
            loss = loss_risk + 0.5 * loss_tts
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= max(len(batch.x_train), 1)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, y_risk, y_tts in val_loader:
                if len(xb) == 0:
                    continue
                xb = xb.to(device)
                y_risk = y_risk.to(device)
                y_tts = y_tts.to(device)
                risk_logits, tts_pred = model(xb)
                loss_risk = risk_criterion(risk_logits, y_risk)
                mask = y_risk > 0.5
                if mask.any():
                    loss_tts = tts_criterion(tts_pred[mask], y_tts[mask]).mean()
                else:
                    loss_tts = torch.tensor(0.0, device=device)
                val_loss += (loss_risk + 0.5 * loss_tts).item() * len(xb)
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
        "max_time_to_stall_s": max_tts,
        "history": history,
        "risk_threshold": 0.5,
    }
    with open(model_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta
