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
    y_risk_train: np.ndarray
    y_tts_train: np.ndarray
    x_val: np.ndarray
    y_risk_val: np.ndarray
    y_tts_val: np.ndarray
    x_test: np.ndarray
    y_risk_test: np.ndarray
    y_tts_test: np.ndarray
    feature_columns: list[str]


class StallSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y_risk: np.ndarray, y_tts: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_risk = torch.tensor(y_risk, dtype=torch.float32)
        self.y_tts = torch.tensor(y_tts, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_risk[idx], self.y_tts[idx]


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
    risk_col: str = "stall_risk",
    tts_col: str = "time_to_stall_s",
    session_col: str | None = "session_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    xs, y_risk, y_tts, indices = [], [], [], []
    groups = df.groupby(session_col, sort=False) if session_col and session_col in df.columns else [(0, df)]

    for _, group in groups:
        values = group[feature_columns].values.astype(np.float32)
        risk = group[risk_col].values.astype(np.float32)
        tts = group[tts_col].values.astype(np.float32)
        phase = group["stall_phase"].values.astype(np.int32) if "stall_phase" in group.columns else np.zeros(len(group))

        for i in range(seq_len, len(group)):
            if phase[i] == 1:
                continue
            xs.append(values[i - seq_len : i])
            y_risk.append(risk[i])
            y_tts.append(max(tts[i], 0.0))
            indices.append(int(group.index[i]))

    if not xs:
        return (
            np.empty((0, seq_len, len(feature_columns))),
            np.empty((0,)),
            np.empty((0,)),
            [],
        )
    return np.stack(xs), np.array(y_risk), np.array(y_tts), indices


def split_sequences_chronological(
    df: pd.DataFrame,
    feature_columns: list[str],
    cfg: ModelConfig | None = None,
) -> SequenceBatch:
    cfg = cfg or ModelConfig()
    x, yr, yt, _ = build_sequences(df, feature_columns, cfg.sequence_length)
    if len(x) == 0:
        raise ValueError("Not enough samples to build sequences.")

    n = len(x)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    return SequenceBatch(
        x_train=x[:train_end],
        y_risk_train=yr[:train_end],
        y_tts_train=yt[:train_end],
        x_val=x[train_end:val_end],
        y_risk_val=yr[train_end:val_end],
        y_tts_val=yt[train_end:val_end],
        x_test=x[val_end:],
        y_risk_test=yr[val_end:],
        y_tts_test=yt[val_end:],
        feature_columns=feature_columns,
    )


def split_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    cfg: ModelConfig | None = None,
    random_seed: int = 42,
) -> SequenceBatch:
    n_sessions = df["session_id"].nunique() if "session_id" in df.columns else 1
    if n_sessions <= 1:
        return split_sequences_shuffled(df, feature_columns, cfg, random_seed)
    return split_sequences_by_session(df, feature_columns, cfg, random_seed)


def split_sequences_shuffled(
    df: pd.DataFrame,
    feature_columns: list[str],
    cfg: ModelConfig | None = None,
    random_seed: int = 42,
) -> SequenceBatch:
    cfg = cfg or ModelConfig()
    x, yr, yt, _ = build_sequences(df, feature_columns, cfg.sequence_length)
    if len(x) == 0:
        raise ValueError("Not enough samples to build sequences.")

    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(x))
    x, yr, yt = x[order], yr[order], yt[order]

    n = len(x)
    train_end = int(n * cfg.train_ratio)
    val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    return SequenceBatch(
        x_train=x[:train_end],
        y_risk_train=yr[:train_end],
        y_tts_train=yt[:train_end],
        x_val=x[train_end:val_end],
        y_risk_val=yr[train_end:val_end],
        y_tts_val=yt[train_end:val_end],
        x_test=x[val_end:],
        y_risk_test=yr[val_end:],
        y_tts_test=yt[val_end:],
        feature_columns=feature_columns,
    )


def split_sequences_by_session(
    df: pd.DataFrame,
    feature_columns: list[str],
    cfg: ModelConfig | None = None,
    random_seed: int = 42,
) -> SequenceBatch:
    cfg = cfg or ModelConfig()
    if "session_id" not in df.columns:
        raise ValueError("session_id required for session-based split")

    rng = np.random.default_rng(random_seed)

    def _split_ids(session_ids: list[int]) -> tuple[set[int], set[int], set[int]]:
        ids = list(session_ids)
        rng.shuffle(ids)
        n = len(ids)
        if n == 0:
            return set(), set(), set()
        if n == 1:
            return {ids[0]}, set(), set()
        if n == 2:
            return {ids[0]}, set(), {ids[1]}
        n_train = max(1, int(n * cfg.train_ratio))
        n_val = max(1, int(n * cfg.val_ratio))
        return set(ids[:n_train]), set(ids[n_train : n_train + n_val]), set(ids[n_train + n_val :])

    if "file_label" in df.columns:
        normal_ids = df.loc[df["file_label"] == "normal", "session_id"].unique().tolist()
        stall_ids = df.loc[df["file_label"] == "stalled", "session_id"].unique().tolist()
        tr_n, va_n, te_n = _split_ids(normal_ids)
        tr_s, va_s, te_s = _split_ids(stall_ids)
        train_ids = tr_n | tr_s
        val_ids = va_n | va_s
        test_ids = te_n | te_s
    else:
        sessions = sorted(df["session_id"].unique())
        train_ids, val_ids, test_ids = _split_ids(sessions)

    def _pack(session_ids: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not session_ids:
            return np.empty((0, cfg.sequence_length, len(feature_columns))), np.array([]), np.array([])
        part = df[df["session_id"].isin(session_ids)]
        x, yr, yt, _ = build_sequences(part, feature_columns, cfg.sequence_length)
        return x, yr, yt

    x_tr, yr_tr, yt_tr = _pack(train_ids)
    x_va, yr_va, yt_va = _pack(val_ids)
    x_te, yr_te, yt_te = _pack(test_ids)

    if len(x_tr) == 0:
        raise ValueError("Not enough training sequences. Add more recordings.")

    return SequenceBatch(
        x_train=x_tr,
        y_risk_train=yr_tr,
        y_tts_train=yt_tr,
        x_val=x_va,
        y_risk_val=yr_va,
        y_tts_val=yt_va,
        x_test=x_te,
        y_risk_test=yr_te,
        y_tts_test=yt_te,
        feature_columns=feature_columns,
    )


def make_dataloader(
    x: np.ndarray,
    y_risk: np.ndarray,
    y_tts: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(StallSequenceDataset(x, y_risk, y_tts), batch_size=batch_size, shuffle=shuffle)
