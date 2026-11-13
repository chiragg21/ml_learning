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

# Training-recipe improvement round, applied to v1 (the best of the
# four architecture-comparison runs along with v3 - see VERSIONS.md/
# LEARNING.md and the conversation history for the v0-v3 comparison).
# Identical to translation_compare_v1.py in every way EXCEPT the
# training recipe itself, so any difference in the result is
# attributable to these two changes alone, not a confound with
# architecture/data/tokenizer:
#
# 1. scheduler="cosine_warmup" instead of "linear_warmup". The old
#    "linear_warmup" scheduler's lr_lambda is `min(1.0, step/warmup)` -
#    after warmup it stays pinned at the peak LR for the rest of
#    training, with NO decay at all. cosine_warmup (already implemented
#    in Trainer._create_scheduler, just unused until now) decays LR to
#    0 over the remaining steps after warmup - the standard recipe for
#    training transformers from scratch.
# 2. training.label_smoothing=0.1, the standard NMT value (used by the
#    original "Attention Is All You Need" transformer). Smooths the
#    cross-entropy target so the model doesn't get overconfident on a
#    single "correct" token - applies only to the training loss;
#    validation loss (Evaluator) is deliberately left as plain
#    cross-entropy so val_loss stays comparable across runs with
#    different label_smoothing values.
#
# Also benefits "for free" from Trainer's best-by-val-loss checkpoint
# fix (saves whichever epoch actually validated best, not just
# whatever the last epoch produced) - the earlier v1 run predates that
# fix, so this is also the first v1 run to actually use it.
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
                "d_model": 384,
                "num_heads": 8,
                "d_ff": 1536,
                "num_layers": 4,
                "dropout_param": 0.1,
            },
        ),
        decoder=ComponentConfig(
            name="decoder",
            version="v1",
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
        extra={"bucket_by_length": True},
    ),
    training=TrainingConfig(
        epochs=5,
        batch_size=128,
        learning_rate=3e-4,
        scheduler="cosine_warmup",
        warmup_steps=250,
        save_every_n_steps=500,
        mixed_precision=True,
        precision="fp16",
        label_smoothing=0.1,
    ),
    evaluation=EvaluationConfig(),
    runtime=RuntimeConfig(
        num_workers=2,
        # Still off - see translation_compare_v1.py's comment on the
        # CUDA-specific Inductor stall this hit on a real T4.
        compile_model=False,
        extra={"prefetch_factor": 4},
    ),
)
