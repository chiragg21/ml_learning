from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvironmentInfo:
    python_version: str
    pytorch_version: str
    cuda_version: str | None
    device: str
    device_name: str | None
    operating_system: str
    cpu: str
    cpu_count: int | None
