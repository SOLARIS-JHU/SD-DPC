"""Neural-network baseline policy (NN-DPC) for comparison against the sparse policy.

A bounded MLP mapping ``[x, r] -> u`` with hard input limits, wrapped as a Neuromancer
Node so it plugs into the same DPC training graph as the sparse policy (Sec. 5 baselines).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import yaml

import torch
from neuromancer.modules import blocks
from neuromancer.modules.activations import activations
from neuromancer.system import Node

__all__ = [
    "NeuralPolicy", "build_nn_policy", "load_nn_policy", "nn_checkpoint_loss_match"
]


class NeuralPolicy(torch.nn.Module):
    """Callable ``(x, r) -> u`` wrapper around a trained Neuromancer policy node."""

    def __init__(self, node: Node):
        super().__init__()
        self.node = node

    def forward(self, x, r):
        return self.node.callable(torch.cat([x, r], dim=-1))


def build_nn_policy(system, cfg: Dict) -> Node:
    """Build a bounded-MLP policy node ``['xn','r'] -> ['u']``."""
    layers = cfg.get("layers", [32, 32])
    nonlin = activations[cfg.get("activation", "relu")]
    net = blocks.MLP_bounds(
        insize=system.nx + system.nx, outsize=system.nu, hsizes=layers,
        nonlin=nonlin, min=system.umin, max=system.umax,
    )
    return Node(net, ["xn", "r"], ["u"], name="neural_policy")


def nn_checkpoint_loss_match(system, path, reference_cfg: Dict) -> Optional[bool]:
    """Check a modern NN checkpoint snapshot against the SD-DPC loss reference.

    Returns ``None`` for legacy checkpoints that have no adjacent config snapshot.
    """
    path = Path(path)
    run_dir = next((parent for parent in path.parents if parent.name.startswith("run_")), None)
    if run_dir is None:
        return None
    snapshot = None
    for candidate in (run_dir / "config.yaml", run_dir / "run_config.json"):
        if candidate.exists():
            with open(candidate, "r") as handle:
                snapshot = yaml.safe_load(handle) if candidate.suffix == ".yaml" else json.load(handle)
            break
    if snapshot is None:
        return None

    expected = system.dpc_loss_spec(reference_cfg)
    actual_weights = snapshot.get("weights", snapshot)
    if int(snapshot.get("nsteps", -1)) != int(expected["horizon"]):
        return False
    for key, value in expected["weights"].items():
        if key not in actual_weights or float(actual_weights[key]) != float(value):
            return False
    return True


def load_nn_policy(system, path, cfg: Dict, device=None) -> NeuralPolicy:
    """Rebuild the NN-DPC architecture and load its saved Neuromancer state dict."""
    device = device or torch.device("cpu")
    node = build_nn_policy(system, cfg.get("nn_policy", cfg))
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Older torch releases do not expose ``weights_only``.
        state = torch.load(path, map_location=device)
    node.load_state_dict(state, strict=True)
    policy = NeuralPolicy(node).to(device).eval()
    if "weights" in cfg:
        match = nn_checkpoint_loss_match(system, path, cfg)
        policy.dpc_loss_reference_verified = match
        if match is not True:
            reason = "has no training-config snapshot" if match is None else "uses a different loss/horizon"
            message = f"NN-DPC checkpoint {path} {reason}; retrain with policy.yaml for strict parity"
            if bool(cfg.get("require_matched_nn_checkpoint", False)):
                raise ValueError(message)
            warnings.warn(message)
    return policy
