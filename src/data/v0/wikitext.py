from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WikitextParameters:
    train_file: str = "data/raw/wikitext-2/train.txt"
    validation_file: str = "data/raw/wikitext-2/validation.txt"

    # Each pair is a sliding window split into (context, continuation):
    # the encoder reads `context_len` characters, the decoder learns to
    # generate the `continuation_len` characters that follow it.
    context_len: int = 128
    continuation_len: int = 64
    stride: int = 96

    # Window generation over the full corpus produces far more pairs than
    # are needed for a from-scratch run; cap and shuffle instead of always
    # training on the same leading slice of the file.
    max_train_pairs: int | None = 20000
    max_val_pairs: int | None = 1000
    seed: int = 42


def _is_heading(line: str) -> bool:
    return (
        line.startswith("=")
        and line.endswith("=")
    )


def _load_clean_text(path: str | Path) -> str:
    """
    Wikitext-2-raw ships one paragraph/heading per line, with blank lines
    between them and section headings wrapped in `=` (e.g. `= Title =`,
    `= = Subsection = =`). Drop both so the corpus is just running prose.
    """

    lines = []

    with Path(path).open("r", encoding="utf-8") as file:

        for raw_line in file:

            line = raw_line.strip()

            if not line or _is_heading(line):
                continue

            lines.append(line)

    return "\n".join(lines)


def _make_windows(
    text: str,
    context_len: int,
    continuation_len: int,
    stride: int,
) -> list[tuple[str, str]]:

    window_len = context_len + continuation_len

    pairs = []

    for start in range(0, len(text) - window_len + 1, stride):

        window = text[start : start + window_len]

        pairs.append((
            window[:context_len],
            window[context_len:],
        ))

    return pairs


def _sample(
    pairs: list[tuple[str, str]],
    max_pairs: int | None,
    rng: random.Random,
) -> list[tuple[str, str]]:

    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs

    return rng.sample(pairs, max_pairs)


def build_wikitext(
    parameters: dict,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], str]:
    """
    Real language-modeling data source: wikitext-2 split into
    (context, continuation) pairs via a sliding window, so the existing
    encoder-decoder pipeline trains on "read this, predict what comes
    next" instead of the toy_copy placeholder's synthetic reversal task.
    """

    config = WikitextParameters(**parameters)
    rng = random.Random(config.seed)

    train_text = _load_clean_text(config.train_file)
    val_text = _load_clean_text(config.validation_file)

    train_pairs = _make_windows(
        train_text,
        config.context_len,
        config.continuation_len,
        config.stride,
    )

    val_pairs = _make_windows(
        val_text,
        config.context_len,
        config.continuation_len,
        config.stride,
    )

    train_pairs = _sample(train_pairs, config.max_train_pairs, rng)
    val_pairs = _sample(val_pairs, config.max_val_pairs, rng)

    return train_pairs, val_pairs, train_text
