"""Build LSTM sequence datasets from processed feature frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig


@dataclass
class SequenceBatch:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_columns: list[str]


class StallSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    seq_len: int,
    session_col: str | None = "session_id",
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    if session_col and session_col in df.columns:
        groups = df.groupby(session_col, sort=False)
    else:
        groups = [(0, df)]

    for _, group in groups:
        values = group[feature_columns].values.astype(np.float32)
        targets = group[target_col].values.astype(np.float32)
        for i in range(seq_len, len(group)):
            xs.append(values[i - seq_len : i])
            ys.append(targets[i])

    if not xs:
        return np.empty((0, seq_len, len(feature_columns))), np.empty((0,))
    return np.stack(xs), np.array(ys)


def split_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "stall_imminent",
    cfg: ModelConfig | None = None,
    random_seed: int = 42,
) -> SequenceBatch:
    cfg = cfg or ModelConfig()
    session_col = "session_id" if "session_id" in df.columns else None

    x_parts, y_parts = [], []
    if session_col:
        for _, group in df.groupby(session_col, sort=False):
            x_g, y_g = build_sequences(group, feature_columns, target_col, cfg.sequence_length, session_col=None)
            if len(x_g):
                x_parts.append(x_g)
                y_parts.append(y_g)
    else:
        x_all, y_all = build_sequences(df, feature_columns, target_col, cfg.sequence_length, session_col=None)
        x_parts, y_parts = [x_all], [y_all]

    x = np.concatenate(x_parts) if x_parts else np.empty((0, cfg.sequence_length, len(feature_columns)))
    y = np.concatenate(y_parts) if y_parts else np.empty((0,))
    if len(x) == 0:
        raise ValueError("Not enough samples to build sequences. Collect more data.")

    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(x))
    x, y = x[order], y[order]

    n = len(x)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    return SequenceBatch(
        x_train=x[:train_end],
        y_train=y[:train_end],
        x_val=x[train_end:val_end],
        y_val=y[train_end:val_end],
        x_test=x[val_end:],
        y_test=y[val_end:],
        feature_columns=feature_columns,
    )


def make_dataloader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = StallSequenceDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
