"""Pipeline configuration for single-motor stall prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NORMAL_MOTOR_TESTS_DIR = BASE_DIR / "normal_motor_tests"
STALL_MOTOR_TESTS_DIR = BASE_DIR / "stall_motor_tests"
STALL_TIMES_PATH = BASE_DIR / "stall_times.json"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "outputs"


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_output_dir(base: Path | None = None) -> Path:
    d = (base or OUTPUT_DIR) / f"run_{make_run_id()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_model_dir(base: Path | None = None) -> Path:
    d = (base or MODEL_DIR) / f"run_{make_run_id()}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class SanitizeConfig:
    max_current_a: float = 5.0
    min_dt_us: int = 100
    max_dt_us: int = 2_000_000
    interpolate_batch_gaps: bool = True


@dataclass
class SpikeConfig:
    spike_multiplier: float = 3.0
    min_spike_a: float = 0.003
    baseline_window_s: float = 0.5


@dataclass
class LabelConfig:
    warning_window_s: float = 5.0
    stall_times_path: Path = field(default_factory=lambda: STALL_TIMES_PATH)


@dataclass
class FeatureConfig:
    resample_hz: float = 100.0
    rolling_window_s: float = 0.2
    # LSTM input: current history only (timestamps used for ordering/labels only)
    model_input_column: str = "current_a"
    feature_columns: list[str] = field(
        default_factory=lambda: ["current_a"]
    )


@dataclass
class ModelConfig:
    sequence_length: int = 100
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 30
    early_stop_patience: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass
class PipelineConfig:
    sanitize: SanitizeConfig = field(default_factory=SanitizeConfig)
    spike: SpikeConfig = field(default_factory=SpikeConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    random_seed: int = 42

    def ensure_dirs(self) -> None:
        for d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR):
            d.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = PipelineConfig()
