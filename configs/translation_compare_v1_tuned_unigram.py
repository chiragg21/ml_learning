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

# Tokenizer-swap experiment, built on translation_compare_v1_tuned.py:
# identical in every way except the tokenizer, so any difference is
# attributable to tokenizer choice alone, isolated from the training-
# recipe changes already tested there.
#
# Switched from bpe_v1 (byte-level BPE) to unigram_v3 (SentencePiece-
# style Unigram), based on a real empirical comparison run on this
# project's actual 500k-pair OPUS-100 corpus (same corpus, same
# vocab_size=8000, measured on the 1419 held-out validation pairs):
#
#   tokenizer      avg_src_tok  avg_tgt_tok  unk_rate  roundtrip_exact
#   bpe_v1              16.69        18.41      0.0%            100.0%
#   wordpiece_v2        17.29        18.63      0.0%             45.4%
#   unigram_v3           16.46        17.55      0.0%            100.0%
#
# unigram_v3 won on every axis that matters: fewest tokens per sentence
# (best compression - shorter effective sequences for the same text),
# zero out-of-vocabulary tokens, and perfect round-trip text
# reproduction (SentencePiece-style whitespace handling, rewriting
# spaces to an explicit "_" marker rather than wordpiece_v2's lossy
# space-rejoining - see tokenizer_v3.py / VERSIONS.md). wordpiece_v2's
# 45% round-trip failure rate is a real, concrete downside: decoded
# text doesn't match the original whitespace/punctuation structure
# more than half the time. unigram_v3's one real cost - ~110s to train
# vs. single-digit seconds for the others - is a one-time setup cost
# per experiment, irrelevant next to a multi-hour training run.
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
        name="unigram",
        version="v3",
        parameters={
            # Matched to bpe_v1's vocab_size for a fair comparison.
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
