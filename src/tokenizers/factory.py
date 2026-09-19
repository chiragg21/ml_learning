from __future__ import annotations

import pickle
from pathlib import Path
from typing import Type

from src.tokenizers.v0.tokenizer_v0 import CharTokenizer


class TokenizerFactory:
    """
    Builds a registered tokenizer version from a generic ExperimentConfig,
    and persists/restores tokenizer instances generically (via pickle) so
    saving/loading works regardless of a given version's internal state.
    """

    _registry: dict[str, Type] = {
        "char_v0": CharTokenizer,
    }

    @classmethod
    def register(
        cls,
        name: str,
        tokenizer_class: Type,
    ) -> None:

        if name in cls._registry:
            raise ValueError(
                f"Tokenizer '{name}' is already registered."
            )

        cls._registry[name] = tokenizer_class

    @classmethod
    def build(
        cls,
        config,
        corpus: str,
    ):

        key = f"{config.tokenizer.name}_{config.tokenizer.version}"

        if key not in cls._registry:
            raise ValueError(
                f"Unknown tokenizer: {key}"
            )

        tokenizer_class = cls._registry[key]

        return tokenizer_class(
            corpus,
            **config.tokenizer.parameters,
        )

    @staticmethod
    def save(
        tokenizer,
        path: str | Path,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open("wb") as file:
            pickle.dump(tokenizer, file)

    @staticmethod
    def load(
        path: str | Path,
    ):

        with Path(path).open("rb") as file:
            return pickle.load(file)
