from __future__ import annotations

from typing import Any, Iterable

# Importing this package runs its auto-discovery, registering every step
# module placed under src/data/pipeline/steps/.
from src.data.pipeline import steps  # noqa: F401
from src.data.pipeline.registry import StepRegistry


class DataPipeline:
    """
    Runs a config-defined sequence of named steps over data, each step's
    output feeding the next. A step never needs to know what came before or
    after it - it just receives whatever the previous step returned (or the
    initial data, for the first step) and returns the next stage's data.
    Works for any data shape (raw text, records, tensors, ...); that's
    entirely up to what each registered step expects and returns.
    """

    def __init__(self, step_configs: Iterable):
        self.step_configs = list(step_configs)

    def run(self, data: Any) -> Any:

        for step_config in self.step_configs:

            step_fn = StepRegistry.get(step_config.name)

            data = step_fn(data, **step_config.parameters)

        return data
