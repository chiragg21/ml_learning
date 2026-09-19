from __future__ import annotations

from typing import Callable

from src.data.v0.toy_copy import build_toy_copy


class DataFactory:
    """
    Registry mapping "{data.name}_{data.version}" to a builder that returns
    (train_pairs, validation_pairs, tokenizer_corpus), where pairs are lists
    of (source_text, target_text) tuples. Register new data sources here;
    the real, configurable data-preparation pipeline is a future iteration.
    """

    _registry: dict[str, Callable] = {
        "toy_copy_v0": build_toy_copy,
    }

    @classmethod
    def register(
        cls,
        name: str,
        builder: Callable,
    ) -> None:

        if name in cls._registry:
            raise ValueError(
                f"Data source '{name}' is already registered."
            )

        cls._registry[name] = builder

    @classmethod
    def build(cls, config):

        key = f"{config.data.name}_{config.data.version}"

        if key not in cls._registry:
            raise ValueError(
                f"Unknown data source: {key}"
            )

        builder = cls._registry[key]

        return builder(config.data.parameters)
