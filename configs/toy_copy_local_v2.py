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

# Same toy_copy task and model size as toy_copy_v2.py, with a local
# attention window of 8 on both self-attention stacks. Sequences run up
# to ~20 tokens, so the window really does cut attention off.
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
    ),
    training=TrainingConfig(
        epochs=5,
        batch_size=32,
        learning_rate=3e-4,
        scheduler="cosine_warmup",
        warmup_steps=50,
    ),
    evaluation=EvaluationConfig(),
    runtime=RuntimeConfig(),
)
