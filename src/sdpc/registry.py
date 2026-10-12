"""Name -> :class:`System` registry so scripts stay example-agnostic.

Scripts take ``--system <name>``; :func:`make_system` resolves it. Register a new example
by adding one entry here (and implementing the :class:`System` subclass).
"""
from __future__ import annotations

from typing import Dict, Type

from .systems import (
    DoubleIntegratorSystem,
    System,
    TwoTankSystem,
    VanDerPolSystem,
)

SYSTEMS: Dict[str, Type[System]] = {
    "double_integrator": DoubleIntegratorSystem,
    "twotank": TwoTankSystem,
    "vanderpol": VanDerPolSystem,
}

__all__ = ["SYSTEMS", "make_system", "available_systems"]


def make_system(name: str, **kwargs) -> System:
    if name not in SYSTEMS:
        raise KeyError(f"unknown system {name!r}; available: {sorted(SYSTEMS)}")
    return SYSTEMS[name](**kwargs)


def available_systems():
    return sorted(SYSTEMS)
