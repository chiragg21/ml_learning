from __future__ import annotations

from typing import Callable


class StepRegistry:
    """
    Registry of named data-preparation steps. A step is any callable
    `(data, **parameters) -> data`. Register one with the decorator form
    (`@StepRegistry.register("my_step")`) or the direct form
    (`StepRegistry.register("my_step", my_step_fn)`).
    """

    _registry: dict[str, Callable] = {}

    @classmethod
    def register(
        cls,
        name: str,
        step: Callable | None = None,
        *,
        overwrite: bool = False,
    ):

        if step is not None:
            cls._add(name, step, overwrite)
            return step

        def decorator(fn: Callable) -> Callable:
            cls._add(name, fn, overwrite)
            return fn

        return decorator

    @classmethod
    def _add(
        cls,
        name: str,
        step: Callable,
        overwrite: bool,
    ) -> None:

        if name in cls._registry and not overwrite:
            raise ValueError(
                f"Step '{name}' is already registered."
            )

        cls._registry[name] = step

    @classmethod
    def get(cls, name: str) -> Callable:

        if name not in cls._registry:
            raise ValueError(
                f"Unknown data preparation step: '{name}'. Make sure it's "
                "defined under src/data/pipeline/steps/ and registered "
                "with @StepRegistry.register(\"...\")."
            )

        return cls._registry[name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._registry)
