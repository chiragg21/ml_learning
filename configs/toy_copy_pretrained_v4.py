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

# Same toy_copy task and model as toy_copy_v1.py, with the tokenizer
# swapped from char_v0 to pretrained_v4 - tokenizer choice is independent
# of model version, so only the TokenizerConfig below differs.
CONFIG = ExperimentConfig(
    schema_version="0.1.0",
    model=ModelConfig(
        transformer=ComponentConfig(
            name="transformer",
            version="v1",
        ),
        encoder=ComponentConfig(
            name="encoder",
            version="v1",
            parameters={
                "max_seq_len": 64,
                "d_model": 64,
                "num_heads": 4,
                "d_ff": 256,
                "num_layers": 2,
                "dropout_param": 0.1,
            },
        ),
        decoder=ComponentConfig(
            name="decoder",
            version="v1",
            parameters={
                "max_seq_len": 64,
                "d_model": 64,
                "num_heads": 4,
                "d_ff": 256,
                "num_layers": 2,
                "dropout_param": 0.1,
            },
        ),
    ),
    tokenizer=TokenizerConfig(
        name="pretrained",
        version="v4",
        parameters={
            "source": "gpt2",
            "eos_token": "<|endoftext|>",
        },
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
