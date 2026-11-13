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
# the model section is identical to translation_compare_v0.py/_v1.py/
# _v2.py on purpose.
#
# v3's model section replaces every layer's dense SwiGLU FFN with a
# 4-expert, top-2 Mixture of Experts (num_experts=4, top_k=2, the
# defaults for moe_every and dynamic_depth are left alone - every layer
# gets MoE, dynamic depth stays off since VERSIONS.md/LEARNING.md note
# it doesn't yet save real compute). This roughly *doubles* this
# model's parameter count versus v1/v2 at the same d_model/d_ff (see
# the parameter-count comparison already run for this project), while
# per-token compute stays close to v1's dense FFN cost (top_k=2 experts
# active per token, not all 4).
#
# Also kept GQA (num_kv_heads=2) from translation_compare_v2.py, since
# v3 builds on v2 rather than replacing it - without this, the v2-vs-v3
# comparison would conflate "added MoE" with "also removed GQA".
#
# Honest caveat worth remembering when reading this run's results: MoE's
# extra capacity is not automatically "more task capability" - on this
# single narrow translation task the router will likely just partition
# the one task in various learned ways, not grow genuinely new skills.
# A real demonstration of MoE's value would need a second, separate
# experiment on more diverse data/tasks - this config is for the
# architecture-only comparison, not that follow-up.
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
        # batch_size at this GPU's memory headroom. Note MoE's extra
        # parameters mean this version will use more GPU memory than
        # v0/v1/v2 at the same batch_size - if you hit an OOM here,
        # lowering batch_size is the first lever, not changing anything
        # above (that would break the apples-to-apples comparison).
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
        compile_model=True,
        extra={"prefetch_factor": 4},
    ),
)
