from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def _serialize(value: Any) -> Any:

    if is_dataclass(value):
        return _serialize(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): _serialize(val)
            for key, val in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }

    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]

    if isinstance(value, set):
        return sorted(
            (_serialize(item) for item in value),
            key=str,
        )

    return value


def config_to_dict(config: Any) -> dict:
    return _serialize(config)


def config_to_json(config: Any) -> str:

    return json.dumps(
        config_to_dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def get_config_hash(
    config: Any,
    length: int = 16,
) -> str:

    serialized = config_to_json(config)

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    return digest[:length]