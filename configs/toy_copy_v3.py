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

# Same toy_copy task and model size as toy_copy_v2.py, pointed at
# transformer_v3: every feed-forward layer is a mixture of 4 experts
# with top-2 routing (plus the load-balancing loss), and each layer has a
# depth router that may skip tokens (aiming at ~75% of tokens per layer).
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
                "max_seq_len": 64,
                "d_model": 64,
                "num_heads": 4,
                "num_kv_heads": 2,
                "num_experts": 4,
                "top_k": 2,
                "moe_aux_weight": 0.01,
                "dynamic_depth": True,
                "depth_capacity": 0.75,
                "d_ff": 256,
                "num_layers": 2,
                "dropout_param": 0.1,
            },
        ),
        decoder=ComponentConfig(
            name="decoder",
            version="v3",
            parameters={
                "max_seq_len": 64,
                "d_model": 64,
                "num_heads": 4,
                "num_kv_heads": 2,
                "num_experts": 4,
                "top_k": 2,
                "moe_aux_weight": 0.01,
                "dynamic_depth": True,
                "depth_capacity": 0.75,
                "d_ff": 256,
                "num_layers": 2,
                "dropout_param": 0.1,
            },
        ),
    ),
    tokenizer=TokenizerConfig(
        name="char",
        version="v0",
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
