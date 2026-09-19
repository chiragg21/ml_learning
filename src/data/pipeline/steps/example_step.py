from __future__ import annotations

from src.data.pipeline.registry import StepRegistry

# Template for a new step: add a file here, register a function against a
# name, and reference that name from a DataConfig.steps entry. Delete this
# file once you've added your own steps - it's only a usage example.


@StepRegistry.register("example_identity")
def example_identity(data, **parameters):
    return data
