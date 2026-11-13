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
# apples-to-apples comparison of architecture alone - v1/v2/v3 should
# keep this file's data/tokenizer/training block identical and only
# change the model section (and whatever's new to that version's
# config, e.g. v3's num_experts).
#
# Scaled-up version of translation_v0.py: more data (20k -> 500k pairs),
# a trained byte-level BPE tokenizer instead of the char tokenizer (see
# src/tokenizers/v1/tokenizer_v1.py), and a correspondingly larger model
# (d_model 256->384, d_ff scaled by the same 4x ratio to 1536). Still
# transformer "v0" architecturally (Post-Norm/LayerNorm/GELU/absolute
# position) - only the data/tokenizer/size changed, not the architecture.
#
# filter_by_length below is a coarse, cheap pre-filter on raw character
# count, applied before the tokenizer even exists - it reduces how much
# text gets tokenized/trained on, but it is NOT a correctness guarantee
# on tokenized length: byte-level BPE's base alphabet is UTF-8 bytes, not
# characters, so a multi-byte character (e.g. an accented French letter)
# can occasionally tokenize into *more* tokens than its character count
# if that byte pair wasn't merged by the trained vocabulary. The actual
# max_seq_len guarantee comes from TokenizedSeq2SeqDataset filtering by
# real tokenized length afterward (see scripts/train.py) - this found
# and silently dropped 1 oversized pair out of 50,000 in testing.
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
        # OPUS-100 sentence lengths vary a lot; grouping similar lengths
        # into the same batch means less of each batch is wasted <PAD>
        # tokens, which eases memory pressure and speeds training -
        # complements fp16 below rather than competing with it.
        extra={"bucket_by_length": True},
    ),
    training=TrainingConfig(
        epochs=5,
        # Bumped from 32: GPU memory/utilization headroom observed during
        # a real run was nowhere near the limit at batch_size=32 (a model
        # this size - ~26M params - was never going to be memory-bound),
        # so a bigger batch means better GPU utilization and likely
        # faster wall-clock training, at the cost of fewer total steps.
        #
        # At batch_size=128 in fp32, a real run measured ~11.5GB used on
        # a 16GB T4 - most of that is attention score matrices and other
        # backward-pass activations, not the ~26M model weights
        # themselves. fp16 halves activation memory (recovering headroom
        # instead of spending it) and runs faster on the T4's fp16
        # tensor cores, so it's the next lever, not a bigger batch_size -
        # pushing batch_size further on top of an already-tight budget
        # risks an OOM. Revisit batch_size only after confirming how much
        # headroom fp16 actually frees on real hardware.
        batch_size=128,
        learning_rate=3e-4,
        scheduler="linear_warmup",
        # ~389k pairs / batch_size=128 ~= 3040 steps/epoch * 5 epochs ~=
        # 15200 total steps (down from ~60800 at batch_size=32) - keeping
        # warmup at roughly the same ~1.6% fraction of total training.
        warmup_steps=250,
        save_every_n_steps=500,
        mixed_precision=True,
        precision="fp16",
    ),
    evaluation=EvaluationConfig(),
    runtime=RuntimeConfig(
        num_workers=2,
        compile_model=True,
        extra={"prefetch_factor": 4},
    ),
)
