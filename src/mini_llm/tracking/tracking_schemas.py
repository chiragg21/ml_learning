from dataclasses import dataclass

@dataclass
class RunMetadata:
    experiment_name: str
    run_name: str
    model_version: str
    description: str | None = None
    git_commit: str | None = None

@dataclass
class DatasetConfig:
    name: str
    version: str | None
    train_samples: int
    validation_samples: int
    train_tokens: int
    validation_tokens: int
    vocab_size: int
    sequence_length: int
    source: str
    preprocessing_version: str
    tokenizer_version: str

@dataclass
class ModelConfig:
    architecture: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    vocab_size: int
    context_length: int
    dropout: float
    positional_encoding: str
    normalization: str
    activation: str

@dataclass
class TrainingConfig:
    optimizer: str
    learning_rate: float
    batch_size: int
    epochs: int
    max_steps: int | None
    weight_decay: float
    gradient_clip_norm: float | None
    warmup_steps: int
    scheduler: str | None
    precision: str
    gradient_accumulation_steps: int

@dataclass
class TrainingMetrics:
    train_loss: float
    learning_rate: float
    epoch: int
    step: int
    tokens_seen: int
    training_time_seconds: float

@dataclass
class TrainingStepMetrics:
    step: int
    loss: float
    learning_rate: float
    tokens_seen: int

@dataclass
class TrainingSummary:
    final_train_loss: float
    best_train_loss: float
    training_time_seconds: float
    total_steps: int
    total_tokens: int

@dataclass
class EvaluationMetrics:
    validation_loss: float
    perplexity: float

@dataclass
class ResourceMetrics:
    parameter_count: int
    trainable_parameter_count: int
    model_size_mb: float
    peak_cpu_memory_mb: float | None
    peak_gpu_memory_mb: float | None
    training_time_seconds: float 
    tokens_per_second: float
    # flops: float | None

@dataclass
class InferenceMetrics:
    latency_ms: float
    tokens_per_second: float
    input_tokens: int
    generated_tokens: int
    peak_memory_mb: float

# @dataclass
# class InferenceMetrics:
#     time_to_first_token_ms: float
#     inter_token_latency_ms: float
#     tokens_per_second: float
#     peak_memory_mb: float

@dataclass
class ArtifactInfo:
    model_path: str | None
    tokenizer_path: str | None
    config_path: str | None
    evaluation_path: str | None

@dataclass
class ModelArtifact:
    filename: str
    local_path: str | None
    artifact_uri: str | None
    format: str
    size_bytes: int
    sha256: str | None

@dataclass
class CheckpointInfo:
    path: str
    step: int
    epoch: int
    validation_loss: float
    is_best: bool

@dataclass
class EnvironmentInfo:
    python_version: str
    pytorch_version: str
    cuda_version: str | None
    device: str
    device_name: str | None
    operating_system: str
    cpu: str | None = None
    cpu_count: int | None = None

@dataclass
class ReproducibilityConfig:
    seed: int
    deterministic: bool