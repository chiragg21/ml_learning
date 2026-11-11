from src.datamodels import (
    ComponentConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PipelineStepConfig,
    RuntimeConfig,
    TokenizerConfig,
    TrainingConfig,
)

# context_len/continuation_len below must stay compatible with these:
# encoder max_seq_len >= context_len, decoder max_seq_len >= continuation_len + 1
# (teacher forcing feeds continuation[:-1] into the decoder).
CONTEXT_LEN = 128
CONTINUATION_LEN = 64

CONFIG = ExperimentConfig(
    schema_version="0.1.0",
    model=ModelConfig(
        transformer=ComponentConfig(
            name="transformer",
            version="v0",
        ),
        encoder=ComponentConfig(
            name="encoder",
            version="v0",
            parameters={
                "max_seq_len": CONTEXT_LEN,
                "d_model": 256,
                "num_heads": 8,
                "d_ff": 1024,
                "num_layers": 4,
                "dropout_param": 0.1,
            },
        ),
        decoder=ComponentConfig(
            name="decoder",
            version="v0",
            parameters={
                "max_seq_len": CONTINUATION_LEN + 2,
                "d_model": 256,
                "num_heads": 8,
                "d_ff": 1024,
                "num_layers": 4,
                "dropout_param": 0.1,
            },
        ),
    ),
    tokenizer=TokenizerConfig(
        name="char",
        version="v0",
    ),
    data=DataConfig(
        name="wikitext",
        version="v0",
        parameters={
            "context_len": CONTEXT_LEN,
            "continuation_len": CONTINUATION_LEN,
            "stride": 96,
            "max_train_pairs": 20000,
            "max_val_pairs": 1000,
            "seed": 42,
        },
        steps=(
            PipelineStepConfig(
                name="min_alpha_ratio",
                parameters={"min_ratio": 0.6},
            ),
            PipelineStepConfig(
                name="filter_by_length",
                parameters={"max_len": CONTEXT_LEN},
            ),
        ),
    ),
    training=TrainingConfig(
        epochs=5,
        batch_size=32,
        learning_rate=3e-4,
        scheduler="linear_warmup",
        warmup_steps=200,
        save_every_n_steps=500,
    ),
    evaluation=EvaluationConfig(),
    runtime=RuntimeConfig(),
)
