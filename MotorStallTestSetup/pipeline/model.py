from __future__ import annotations

import torch
import torch.nn as nn


class StallLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_time_to_stall_s: float = 5.0,
    ):
        super().__init__()
        self.max_time_to_stall_s = max_time_to_stall_s
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Linear(hidden_size // 2, 1)
        self.tts_head = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        h = self.shared(out[:, -1, :])
        risk_logits = self.risk_head(h).squeeze(-1)
        tts = torch.sigmoid(self.tts_head(h).squeeze(-1)) * self.max_time_to_stall_s
        return risk_logits, tts
