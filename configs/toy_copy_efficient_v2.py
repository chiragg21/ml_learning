from src.datamodels import (
    ComponentConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    TokenizerConfig,
    TrainingConfig,
)

# Same task and model size as toy_copy_local_v2.py with every v2
# efficiency feature on: banded local attention, packed sequences, bf16
# mixed precision and gradient checkpointing. (torch.compile and the
# streaming/bucketing data options are separate switches, shown below as
# comments - compile adds start-up time that a toy run doesn't repay.)
CONFIG = ExperimentConfig(
    schema_version="0.1.0",
    model=ModelConfig(
        transformer=ComponentConfig(
            name="transformer",
            version="v2",
        ),
        encoder=ComponentConfig(
            name="encoder",
            version="v2",
            parameters={
                "max_seq_len": 64,
                "d_model": 64,
                "num_heads": 4,
                "num_kv_heads": 2,
                "window_size": 8,
                "banded": True,
                "d_ff": 256,
                "num_layers": 2,
                "dropout_param": 0.1,
            },
        ),
        decoder=ComponentConfig(
            name="decoder",
            version="v2",
            parameters={
                "max_seq_len": 64,
                "d_model": 64,
                "num_heads": 4,
                "num_kv_heads": 2,
                "window_size": 8,
                "banded": True,
                "d_ff": 256,
                "num_layers": 2,
                "dropout_param": 0.1,
            },
        ),
    ),
    tokenizer=TokenizerConfig(
        name="char",
        version="v0",
    ),
    data=DataConfig(
        name="toy_copy",
        version="v0",
        parameters={
            "vocab": "abcdefghijklmnopqrstuvwxyz",
            "min_len": 5,
            "max_len": 20,
            "train_size": 2000,
            "val_size": 200,
            "seed": 42,
        },
        extra={
            "pack_sequences": True,
            # "streaming": True, "shuffle_buffer": 1000,   (instead of packing)
            # "bucket_by_length": True,
        },
    ),
    training=TrainingConfig(
        epochs=5,
        batch_size=32,
        learning_rate=3e-4,
        scheduler="cosine_warmup",
        warmup_steps=50,
        mixed_precision=True,
        precision="bf16",
        gradient_checkpointing=True,
    ),
    evaluation=EvaluationConfig(),
    # compile_model=True turns on torch.compile; num_workers > 0 with
    # extra={"prefetch_factor": 4} overlaps data loading with compute.
    runtime=RuntimeConfig(),
)
