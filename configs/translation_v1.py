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

# Same OPUS-100 en-fr data and model size as translation_v0.py - only the
# transformer version and scheduler differ. That's the point: switching
# architecture version (or dataset) is purely a config change here.
MAX_SEQ_LEN = 160

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
                "max_seq_len": MAX_SEQ_LEN,
                "d_model": 256,
                "num_heads": 8,
                "d_ff": 1024,
                "num_layers": 4,
                "dropout_param": 0.1,
            },
        ),
        decoder=ComponentConfig(
            name="decoder",
            version="v1",
            parameters={
                # +2 for the <BOS>/<EOS> the tokenizer adds to targets.
                "max_seq_len": MAX_SEQ_LEN + 2,
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
        name="translation",
        version="v0",
        parameters={
            "data_dir": "data/raw/opus100_en_fr",
            "source_lang": "en",
            "target_lang": "fr",
            "max_train_pairs": 20000,
            "max_val_pairs": 1000,
            "seed": 42,
        },
        steps=(
            PipelineStepConfig(
                name="filter_by_length",
                parameters={"max_len": MAX_SEQ_LEN},
            ),
        ),
    ),
    training=TrainingConfig(
        epochs=5,
        batch_size=32,
        learning_rate=3e-4,
        # v1's roadmap item: linear warmup then cosine decay, instead of
        # v0's plain linear warmup with no decay afterward.
        scheduler="cosine_warmup",
        warmup_steps=200,
        save_every_n_steps=500,
    ),
    evaluation=EvaluationConfig(),
    runtime=RuntimeConfig(),
)
