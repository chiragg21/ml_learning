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

# One of four configs (translation_compare_v0.py .. _v3.py) meant to be
# trained on the *same* data/tokenizer/training budget, one version at a
# time, in separate sessions, so the resulting metrics are a fair
# apples-to-apples comparison of architecture alone. Everything below
# the model section is identical to translation_compare_v0.py on
# purpose - only the model's version changed (v0 -> v1: Pre-Norm,
# RMSNorm, RoPE, SwiGLU, weight tying - see VERSIONS.md/LEARNING.md).
# No new config fields are needed for v1 beyond what v0 already had;
# rope_theta defaults to the standard 10000.0.
#
# filter_by_length below is a coarse, cheap pre-filter on raw character
# count, applied before the tokenizer even exists - it reduces how much
# text gets tokenized/trained on, but it is NOT a correctness guarantee
# on tokenized length: byte-level BPE's base alphabet is UTF-8 bytes, not
# characters, so a multi-byte character (e.g. an accented French letter)
# can occasionally tokenize into *more* tokens than its character count
# if that byte pair wasn't merged by the trained vocabulary. The actual
# max_seq_len guarantee comes from TokenizedSeq2SeqDataset filtering by
# real tokenized length afterward (see scripts/train.py).
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
        # OPUS-100 sentence lengths vary a lot; grouping similar lengths
        # into the same batch means less of each batch is wasted <PAD>
        # tokens, which eases memory pressure and speeds training -
        # complements fp16 below rather than competing with it.
        extra={"bucket_by_length": True},
    ),
    training=TrainingConfig(
        epochs=5,
        # Kept identical to translation_compare_v0.py on purpose - see
        # that file's comment for why 128/fp16 was chosen over a bigger
        # batch_size at this GPU's memory headroom.
        batch_size=128,
        learning_rate=3e-4,
        scheduler="linear_warmup",
        warmup_steps=250,
        save_every_n_steps=500,
        mixed_precision=True,
        precision="fp16",
    ),
    evaluation=EvaluationConfig(),
    runtime=RuntimeConfig(
        num_workers=2,
        # torch.compile disabled: on this T4, torch._inductor hit
        # "Not enough SMs to use max_autotune_gemm mode" and then
        # stalled indefinitely (GPU memory near-zero, no training
        # progress for 12+ minutes, reproduced across two restarts).
        # This is a CUDA-specific Inductor stall that can't be caught
        # by testing on CPU-only dev hardware - don't re-enable
        # without verifying on a real GPU first. fp16 + bucketing +
        # parallel workers are the optimizations that actually help.
        compile_model=False,
        extra={"prefetch_factor": 4},
    ),
)
