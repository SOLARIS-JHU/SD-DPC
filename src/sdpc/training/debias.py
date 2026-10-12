"""Configuration for fixed-support sparse coefficient refitting."""

from __future__ import annotations

import copy
from typing import Dict


def make_debias_config(cfg: Dict) -> Dict:
    """Start a fresh AdamW phase with the selected support and no L1 or pruning."""
    debias = cfg.get("debias", {})
    if not debias.get("enabled", False):
        raise ValueError("debias.enabled must be true to build a debias config")
    epochs = int(debias["epochs"])
    if epochs < 1:
        raise ValueError("debias.epochs must be positive")
    if debias.get("learning_rate", "initial_training_rate") != "initial_training_rate":
        raise ValueError("Only the initial training rate is supported for debiasing")
    result = copy.deepcopy(cfg)
    result["debias"]["enabled"] = False
    result["epochs"] = epochs
    result["weights"]["l1_coef"] = 0.0
    result["proximal_l1"] = False
    result["l1_decay"] = 1.0
    result["max_unchange"] = 0
    result["prune_noise"] = 0.0
    result["prune_every"] = epochs + 1
    result["prune_every_min"] = epochs + 1
    result["change_prune_every"] = False
    result["lr_decay_enabled"] = False
    return result
