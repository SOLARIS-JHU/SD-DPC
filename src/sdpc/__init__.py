"""SparseDPC framework (``sdpc``): modular, reproducible Sparse Differentiable Predictive Control.

A plug-and-play sparse-control pipeline organized into reusable stages:

* :mod:`sdpc.systems`    — system interfaces for the active case studies.
* :mod:`sdpc.sindy`      — the candidate library and SINDy model (dynamics *and* policy).
* :mod:`sdpc.training`   — ADAM-SINDy system-ID (Alg. 1) and sparse-policy DPC learning.
* :mod:`sdpc.safety`     — state and action constraint specifications.
* :mod:`sdpc.baselines`  — MPC (CasADi).
* :mod:`sdpc.eval`       — metrics and the cross-method comparison pipeline.
* :mod:`sdpc.plotting`   — trajectories, coefficient heatmaps.
* :mod:`sdpc.config` / :mod:`sdpc.io` — YAML configs and reproducible run directories.

New systems are added by subclassing :class:`sdpc.systems.System` and registering them in
:mod:`sdpc.registry`; the scripts and notebooks then work unchanged.
"""
from . import (
    baselines,
    config,
    data,
    eval,
    io,
    plotting,
    registry,
    safety,
    sindy,
    systems,
    training,
)
from .registry import SYSTEMS, make_system, available_systems

__all__ = [
    "baselines",
    "config",
    "data",
    "eval",
    "io",
    "plotting",
    "registry",
    "safety",
    "sindy",
    "systems",
    "training",
    "SYSTEMS",
    "make_system",
    "available_systems",
]

__version__ = "0.1.0"
