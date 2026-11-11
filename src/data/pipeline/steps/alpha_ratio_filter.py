from __future__ import annotations

from src.data.pipeline.registry import StepRegistry


def _alpha_ratio(text: str) -> float:

    if not text:
        return 0.0

    alpha_count = sum(
        character.isalpha()
        for character in text
    )

    return alpha_count / len(text)


@StepRegistry.register("min_alpha_ratio")
def min_alpha_ratio(data, min_ratio: float = 0.5):
    """
    Drops pairs where either side is mostly non-alphabetic (punctuation,
    digits, wikitext detokenization artifacts like "@-@"). Useful for
    windowed text sources where a sliding window can land entirely on
    junk rather than prose.
    """

    return [
        (source, target)
        for source, target in data
        if _alpha_ratio(source) >= min_ratio
        and _alpha_ratio(target) >= min_ratio
    ]
