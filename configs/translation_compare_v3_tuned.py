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

# Training-recipe improvement round, applied to v3 (the best of the
# four architecture-comparison runs along with v1 - see VERSIONS.md/
# LEARNING.md and the conversation history for the v0-v3 comparison).
# Identical to translation_compare_v3.py in every way EXCEPT the
# training recipe itself - see translation_compare_v1_tuned.py's
# comment for the full reasoning behind both changes:
#
# 1. scheduler="cosine_warmup" instead of "linear_warmup" (decays LR to
#    0 after warmup, instead of staying pinned at the peak forever).
# 2. training.label_smoothing=0.1, the standard NMT value. Applies
#    only to the training loss, not validation (Evaluator stays plain
#    cross-entropy, so val_loss remains comparable across runs).
#
# v3 is also the first run here to actually exercise Trainer's
# best-by-val-loss checkpoint fix together with MoE's aux_loss and
# GQA - the earlier v3 run predates that fix.
MAX_SEQ_LEN = 160

CONFIG = ExperimentConfig(
    schema_version="0.1.0",
    model=ModelConfig(
        transformer=ComponentConfig(
            name="transformer",
            version="v3",
        ),
        encoder=ComponentConfig(
            name="encoder",
            version="v3",
            parameters={
                "max_seq_len": MAX_SEQ_LEN,
                "d_model": 384,
                "num_heads": 8,
                "d_ff": 1536,
                "num_layers": 4,
                "dropout_param": 0.1,
                "num_kv_heads": 2,
                "num_experts": 4,
                "top_k": 2,
            },
        ),
        decoder=ComponentConfig(
            name="decoder",
            version="v3",
            parameters={
                "max_seq_len": MAX_SEQ_LEN + 2,
                "d_model": 384,
                "num_heads": 8,
                "d_ff": 1536,
                "num_layers": 4,
                "dropout_param": 0.1,
                "num_kv_heads": 2,
                "num_experts": 4,
                "top_k": 2,
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
        # Still off - see translation_compare_v3.py's comment on the
        # CUDA-specific Inductor stall this hit on a real T4.
        compile_model=False,
        extra={"prefetch_factor": 4},
    ),
)
