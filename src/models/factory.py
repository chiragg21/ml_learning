from __future__ import annotations

from typing import Callable, Type

from src.models.template import BaseTransformer
from src.models.v0.datamodels_v0 import (
    DecoderConfigV0,
    EncoderConfigV0,
    TransformerConfigV0,
)
from src.models.v0.transformer_v0 import TransformerV0
from src.models.v1.datamodels_v1 import (
    DecoderConfigV1,
    EncoderConfigV1,
    TransformerConfigV1,
)
from src.models.v1.transformer_v1 import TransformerV1
from src.models.v2.datamodels_v2 import (
    DecoderConfigV2,
    EncoderConfigV2,
    TransformerConfigV2,
)
from src.models.v2.transformer_v2 import TransformerV2


def _build_transformer_v0_config(config, tokenizer=None) -> TransformerConfigV0:

    encoder_params = dict(config.model.encoder.parameters)
    decoder_params = dict(config.model.decoder.parameters)

    if tokenizer is not None:
        encoder_params.setdefault("vocab_size", tokenizer.vocab_size)
        decoder_params.setdefault("target_vocab_size", tokenizer.vocab_size)

    return TransformerConfigV0(
        encoder_config=EncoderConfigV0(**encoder_params),
        decoder_config=DecoderConfigV0(**decoder_params),
    )


def _build_transformer_v1_config(config, tokenizer=None) -> TransformerConfigV1:

    encoder_params = dict(config.model.encoder.parameters)
    decoder_params = dict(config.model.decoder.parameters)

    if tokenizer is not None:
        encoder_params.setdefault("vocab_size", tokenizer.vocab_size)
        decoder_params.setdefault("target_vocab_size", tokenizer.vocab_size)

    return TransformerConfigV1(
        encoder_config=EncoderConfigV1(**encoder_params),
        decoder_config=DecoderConfigV1(**decoder_params),
    )


def _build_transformer_v2_config(config, tokenizer=None) -> TransformerConfigV2:

    encoder_params = dict(config.model.encoder.parameters)
    decoder_params = dict(config.model.decoder.parameters)

    if tokenizer is not None:
        encoder_params.setdefault("vocab_size", tokenizer.vocab_size)
        decoder_params.setdefault("target_vocab_size", tokenizer.vocab_size)

    return TransformerConfigV2(
        encoder_config=EncoderConfigV2(**encoder_params),
        decoder_config=DecoderConfigV2(**decoder_params),
    )


class ModelFactory:
    """
    Builds a registered transformer version from a generic ExperimentConfig.

    Each registry entry maps a "{transformer.name}_{transformer.version}" key
    to the model class plus a builder that adapts the generic ModelConfig
    (transformer/encoder/decoder components) into that version's own config
    dataclass. Register new versions here without touching src/models/vN/.
    """

    _registry: dict[str, tuple[Type[BaseTransformer], Callable]] = {
        "transformer_v0": (TransformerV0, _build_transformer_v0_config),
        "transformer_v1": (TransformerV1, _build_transformer_v1_config),
        "transformer_v2": (TransformerV2, _build_transformer_v2_config),
    }

    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[BaseTransformer],
        config_builder: Callable,
    ) -> None:

        if name in cls._registry:
            raise ValueError(
                f"Model '{name}' is already registered."
            )

        cls._registry[name] = (model_class, config_builder)

    @classmethod
    def create(
        cls,
        config,
        tokenizer=None,
    ) -> BaseTransformer:

        model_name = config.model.transformer.name
        model_version = config.model.transformer.version

        key = f"{model_name}_{model_version}"

        if key not in cls._registry:
            raise ValueError(
                f"Unknown model architecture: {key}"
            )

        model_class, config_builder = cls._registry[key]

        model_config = config_builder(
            config,
            tokenizer=tokenizer,
        )

        return model_class(model_config)
