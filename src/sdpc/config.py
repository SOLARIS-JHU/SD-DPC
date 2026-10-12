"""YAML configuration loading, seeding, and run-directory helpers.

Every stage script loads a YAML file from ``<Example>/configs/<stage>.yaml`` and passes
the parsed dict on to the framework. Configs may include an ``inherit`` key (a path,
relative to the config file, to a base config that is merged underneath it) so that,
e.g., ``safe.yaml`` can inherit shared keys from ``policy.yaml``.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

__all__ = ["load_config", "seed_everything", "deep_update"]


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config, resolving a single-level ``inherit:`` base if present."""
    path = Path(path)
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    inherit = cfg.pop("inherit", None)
    if inherit is not None:
        base_path = (path.parent / inherit).resolve()
        base = load_config(base_path)
        cfg = deep_update(base, cfg)

    cfg["_config_path"] = str(path)
    return cfg


def seed_everything(seed: Optional[int]) -> None:
    """Seed python, numpy and torch RNGs for reproducible runs."""
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
