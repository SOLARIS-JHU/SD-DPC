"""System definitions and the shared :class:`System` interface."""
from .base import System, rk4_step
from .double_integrator import DoubleIntegratorSystem
from .twotank import TwoTankSystem
from .vanderpol import (
    VanDerPolSystem,
    VanDerPolControl,
)

__all__ = [
    "System",
    "rk4_step",
    "DoubleIntegratorSystem",
    "TwoTankSystem",
    "VanDerPolSystem",
    "VanDerPolControl",
]
