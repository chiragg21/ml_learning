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

# Scaled-up version of translation_v0.py: more data (20k -> 500k pairs),
# a trained byte-level BPE tokenizer instead of the char tokenizer (see
# src/tokenizers/v1/tokenizer_v1.py), and a correspondingly larger model
# (d_model 256->384, d_ff scaled by the same 4x ratio to 1536). Still
# transformer "v0" architecturally (Post-Norm/LayerNorm/GELU/absolute
# position) - only the data/tokenizer/size changed, not the architecture.
#
# BPE tokens cover multiple characters each, so the same MAX_SEQ_LEN
# (measured in characters, via filter_by_length below, which runs
# before tokenization) comfortably covers sequences that are now
# *shorter* in token count than the old char-tokenized ones were - this
# is a safe upper bound, not a tight one.
MAX_SEQ_LEN = 160

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
                "max_seq_len": MAX_SEQ_LEN,
                "d_model": 384,
                "num_heads": 8,
                "d_ff": 1536,
                "num_layers": 4,
                "dropout_param": 0.1,
            },
        ),
        decoder=ComponentConfig(
            name="decoder",
            version="v0",
            parameters={
                "max_seq_len": MAX_SEQ_LEN + 2,
                "d_model": 384,
                "num_heads": 8,
                "d_ff": 1536,
                "num_layers": 4,
                "dropout_param": 0.1,
            },
        ),
    ),
    tokenizer=TokenizerConfig(
        name="bpe",
        version="v1",
        parameters={
            # Proportional to this model's size, not a pretrained
            # tokenizer's 30k+ vocab sized for a much bigger model.
            "vocab_size": 8000,
        },
    ),
    data=DataConfig(
        name="translation",
        version="v0",
        parameters={
            "data_dir": "data/raw/opus100_en_fr",
            "source_lang": "en",
            "target_lang": "fr",
            "max_train_pairs": 500000,
            "max_val_pairs": 2000,
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
        scheduler="linear_warmup",
        # More total steps than translation_v0.py (25x the data) - a
        # longer warmup is proportionally still a small fraction of training.
        warmup_steps=1000,
        save_every_n_steps=1000,
    ),
    evaluation=EvaluationConfig(),
    runtime=RuntimeConfig(),
)
