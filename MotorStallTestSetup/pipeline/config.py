"""Pipeline configuration for single-motor stall prediction."""

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MOTOR_DATA_TESTS_DIR = BASE_DIR / "motor_data_tests"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "outputs"


@dataclass
class SanitizeConfig:
    max_current_a: float = 5.0
    min_dt_us: int = 100
    max_dt_us: int = 2_000_000
    interpolate_batch_gaps: bool = True


@dataclass
class SpikeConfig:
    """Spike detection; first spike per session is always ignored (motor startup)."""
    spike_multiplier: float = 3.0
    min_spike_a: float = 0.003
    baseline_window_s: float = 0.5


@dataclass
class LabelConfig:
    """Rule-based stall / precursor labeling."""
    spike_burst_window_s: float = 2.0
    spike_burst_count: int = 2
    trend_window_s: float = 1.0
    trend_slope_threshold: float = 0.005
    prediction_horizon_s: float = 1.0
    stall_sustain_s: float = 0.3
    stall_current_threshold_a: float = 0.02


@dataclass
class FeatureConfig:
    resample_hz: float = 100.0
    rolling_window_s: float = 0.2
    feature_columns: list[str] = field(
        default_factory=lambda: [
            "current_a",
            "d_current",
            "rolling_mean",
            "rolling_std",
            "spike_flag",
            "spike_count_window",
            "trend_slope",
        ]
    )


@dataclass
class ModelConfig:
    sequence_length: int = 50
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
