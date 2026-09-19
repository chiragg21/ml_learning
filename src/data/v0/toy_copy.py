from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class ToyCopyParameters:
    vocab: str = "abcdefghijklmnopqrstuvwxyz"
    min_len: int = 5
    max_len: int = 20
    train_size: int = 2000
    val_size: int = 200
    seed: int = 42


def _generate_pairs(
    parameters: ToyCopyParameters,
    count: int,
    rng: random.Random,
) -> list[tuple[str, str]]:

    pairs = []

    for _ in range(count):

        length = rng.randint(
            parameters.min_len,
            parameters.max_len,
        )

        source = "".join(
            rng.choice(parameters.vocab)
            for _ in range(length)
        )

        pairs.append((source, source[::-1]))

    return pairs


def build_toy_copy(
    parameters: dict,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
    """
    Synthetic sequence-reversal task (target = reversed source), used only
    to validate the training pipeline end-to-end. Not a real data source.
    """

    config = ToyCopyParameters(**parameters)
    rng = random.Random(config.seed)

    train_pairs = _generate_pairs(config, config.train_size, rng)
    val_pairs = _generate_pairs(config, config.val_size, rng)

    return train_pairs, val_pairs, config.vocab
