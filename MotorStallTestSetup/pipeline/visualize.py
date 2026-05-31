from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_dashboard(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Motor Stall — Predictive Maintenance Dashboard",
    full_telemetry_df: pd.DataFrame | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    tel = full_telemetry_df if full_telemetry_df is not None else df
    axes[0].plot(tel["time_s"], tel["current_a"], color="#2563eb", linewidth=0.8)
    if "stall_onset_time_s" in df.columns and (df["stall_onset_time_s"] >= 0).any():
        axes[0].axvline(
            df["stall_onset_time_s"].iloc[0],
            color="#dc2626",
            linestyle="--",
            label="Stall onset (label)",
        )
    axes[0].set_ylabel("Current (A)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Input: current telemetry")

    if "stall_risk_label" in df.columns:
        axes[1].fill_between(
            df["time_s"], 0, df["stall_risk_label"], step="pre", alpha=0.35, color="#f59e0b", label="Pre-stall window (label)"
        )
    axes[1].set_ylabel("Label")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Ground truth: approaching stall (before onset only)")

    if "stall_probability" in df.columns:
        axes[2].plot(df["time_s"], df["stall_probability"], color="#dc2626", linewidth=0.9, label="Stall risk")
        axes[2].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        axes[2].set_ylabel("Risk")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title("Output: stall probability")

    if "time_to_stall_predicted_s" in df.columns:
        axes[3].plot(
            df["time_s"],
            df["time_to_stall_predicted_s"],
            color="#7c3aed",
            linewidth=0.9,
            label="Predicted time-to-stall",
        )
        if "time_to_stall_actual_s" in df.columns:
            actual = df["time_to_stall_actual_s"].replace(-1, float("nan"))
            axes[3].plot(df["time_s"], actual, color="#059669", alpha=0.7, linewidth=0.8, label="Actual (label)")
        axes[3].set_ylabel("Seconds")
        axes[3].set_xlabel("Time (s)")
        axes[3].legend(loc="upper right", fontsize=8)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_title("Output: estimated time until stall")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_current_with_predictions(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Motor Current & Stall Prediction",
) -> None:
    plot_dashboard(df, output_path, title)


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
    plt.ylabel("Combined loss (risk + time-to-stall)")
    plt.title("LSTM Training History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
