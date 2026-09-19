from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ComponentConfig:
    name: str
    version: str
    parameters: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    transformer: ComponentConfig
    encoder: ComponentConfig
    decoder: ComponentConfig


@dataclass(frozen=True)
class TokenizerConfig:
    name: str
    version: str
    parameters: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineStepConfig:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataConfig:
    name: str
    version: str
    parameters: dict[str, Any] = field(default_factory=dict)
    steps: tuple[PipelineStepConfig, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int

    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0

    scheduler: str | None = None
    warmup_steps: int = 0

    mixed_precision: bool = False
    precision: str = "fp32"

    gradient_checkpointing: bool = False

    save_every_n_steps: int | None = None

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationConfig:
    metrics: tuple[str, ...] = (
        "loss",
        "perplexity",
        "token_accuracy",
    )

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    num_workers: int = 0
    seed: int = 42

    compile_model: bool = False

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: str

    model: ModelConfig
    tokenizer: TokenizerConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig