"""Visualization helpers for stall prediction pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_current_with_predictions(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Motor Current & Stall Prediction",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(df["time_s"], df["current_a"], color="#2563eb", linewidth=0.8, label="Current (A)")
    if "spike_flag" in df.columns:
        spikes = df[df["spike_flag"] == 1]
        axes[0].scatter(spikes["time_s"], spikes["current_a"], color="#f59e0b", s=8, label="Spike")
    if "startup_spike" in df.columns:
        startup = df[df.get("startup_spike", 0) == 1]
        if len(startup):
            axes[0].scatter(
                startup["time_s"],
                startup["current_a"],
                color="#10b981",
                s=20,
                marker="x",
                label="Startup spike (ignored)",
            )
    axes[0].set_ylabel("Current (A)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    if "stall_probability" in df.columns:
        axes[1].plot(df["time_s"], df["stall_probability"], color="#dc2626", label="Stall probability")
        axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Threshold")
        pred = df[df["stall_predicted"] == 1]
        if len(pred):
            axes[1].scatter(pred["time_s"], pred["stall_probability"], color="#dc2626", s=6)
    elif "stall_imminent" in df.columns:
        axes[1].fill_between(
            df["time_s"],
            0,
            df["stall_imminent"],
            step="pre",
            alpha=0.4,
            color="#dc2626",
            label="Stall imminent (label)",
        )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Stall signal")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_history(history: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("LSTM Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
