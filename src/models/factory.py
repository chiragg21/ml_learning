from __future__ import annotations

from typing import Type

from src.models.template import BaseTransformer
from src.models.v0.transformer_v0 import TransformerV0


class ModelFactory:

    _registry: dict[str, Type[BaseTransformer]] = {
        "transformer_v0": TransformerV0,
    }

    @classmethod
    def register(
        cls,
        name: str,
        model_class: Type[BaseTransformer],
    ) -> None:

        if name in cls._registry:
            raise ValueError(
                f"Model '{name}' is already registered."
            )

        cls._registry[name] = model_class

    @classmethod
    def create(
        cls,
        config,
    ) -> BaseTransformer:

        model_name = config.model.transformer.name
        model_version = config.model.transformer.version

        key = f"{model_name}_{model_version}"

        if key not in cls._registry:
            raise ValueError(
                f"Unknown model architecture: {key}"
            )

        model_class = cls._registry[key]

        return model_class(config)