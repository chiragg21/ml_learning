from __future__ import annotations

from src.data.pipeline.registry import StepRegistry


@StepRegistry.register("filter_by_length")
def filter_by_length(data, max_len: int):
    """
    Drops any (source, target) pair where either side exceeds `max_len`
    characters - a safety net so a run's `max_seq_len` can't be silently
    violated by a data source that doesn't enforce it itself.
    """

    return [
        (source, target)
        for source, target in data
        if len(source) <= max_len and len(target) <= max_len
    ]
