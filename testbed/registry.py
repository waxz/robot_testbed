"""Plugin registry: sensors, command sources and controllers register themselves by type name."""
from __future__ import annotations

from typing import Any, Callable

REGISTRY: dict[str, dict[str, type]] = {"sensor": {}, "command": {}, "controller": {}}


def register(kind: str, name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        if name in REGISTRY[kind]:
            raise ValueError(f"{kind} type '{name}' registered twice")
        REGISTRY[kind][name] = cls
        cls.type_name = name
        return cls
    return deco


def build(kind: str, spec: dict, **context: Any):
    """Instantiate a registered component from its config block: {type: name, ...options}."""
    spec = dict(spec)
    name = spec.pop("type")
    if name not in REGISTRY[kind]:
        raise KeyError(f"unknown {kind} type '{name}'; known: {sorted(REGISTRY[kind])}")
    return REGISTRY[kind][name](**spec, **context)
