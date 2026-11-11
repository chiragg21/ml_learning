from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranslationParameters:
    data_dir: str = "data/raw/opus100_en_fr"
    source_lang: str = "en"
    target_lang: str = "fr"

    max_train_pairs: int | None = 20000
    max_val_pairs: int | None = 1000
    seed: int = 42


def _read_lines(path: str | Path) -> list[str]:

    with Path(path).open("r", encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file]


def _load_pairs(
    data_dir: Path,
    split: str,
    source_lang: str,
    target_lang: str,
) -> list[tuple[str, str]]:

    source_lines = _read_lines(
        data_dir / f"{split}.{source_lang}.txt"
    )

    target_lines = _read_lines(
        data_dir / f"{split}.{target_lang}.txt"
    )

    return [
        (source, target)
        for source, target in zip(source_lines, target_lines)
        if source and target
    ]


def _sample(
    pairs: list[tuple[str, str]],
    max_pairs: int | None,
    rng: random.Random,
) -> list[tuple[str, str]]:

    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs

    return rng.sample(pairs, max_pairs)


def build_translation(
    parameters: dict,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
    """
    Real machine-translation data source: line-aligned (source, target)
    sentence pairs from a parallel corpus (default: OPUS-100 en-fr,
    downloaded via notebooks/translation_dataset.ipynb). This is the task
    shape TransformerV0's encoder-decoder/cross-attention was built for -
    unlike wikitext's invented context->continuation framing, source and
    target here are genuinely different, paired sequences.
    """

    config = TranslationParameters(**parameters)
    rng = random.Random(config.seed)

    data_dir = Path(config.data_dir)

    train_pairs = _load_pairs(
        data_dir,
        "train",
        config.source_lang,
        config.target_lang,
    )

    val_pairs = _load_pairs(
        data_dir,
        "validation",
        config.source_lang,
        config.target_lang,
    )

    train_pairs = _sample(train_pairs, config.max_train_pairs, rng)
    val_pairs = _sample(val_pairs, config.max_val_pairs, rng)

    corpus = "\n".join(
        text
        for pair in train_pairs
        for text in pair
    )

    return train_pairs, val_pairs, corpus
