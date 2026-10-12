"""Checkpoint save / load helpers for :class:`SINDyVectorized` models.

The checkpoint format is a plain ``dict`` (torch-saved) with four keys:

``lib_cfg``    kwargs to rebuild the :class:`CompiledFunctionLibrary`.
``model_cfg``  kwargs to rebuild the :class:`SINDyVectorized` skeleton (``n_out`` and,
               for policies, ``policy_name``).
``active_idx`` per-output list of active global term indices (the sparsity pattern).
``state_dict`` the model weights.

This is the same on-disk layout produced by the original notebooks, so checkpoints
trained before the refactor load unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from .library import CompiledFunctionLibrary
from .model import SINDyVectorized

__all__ = ["library_cfg", "save_model", "load_model"]


def library_cfg(lib: CompiledFunctionLibrary) -> dict:
    """Serialise the constructor kwargs of a :class:`CompiledFunctionLibrary`."""
    return dict(
        n_features=lib.n_features,
        n_control=lib.n_control,
        include_bias=lib.include_bias,
        max_degree=lib.max_degree,
        add_sqrt=lib.add_sqrt,
        add_xu=lib.add_xu,
        add_u_prod=lib.add_u_prod,
        add_fourier=lib.add_fourier,
        max_freq=lib.max_freq,
        is_policy=lib.is_policy,
    )


def save_model(model: SINDyVectorized, path: Union[str, Path]) -> Path:
    """Save a (possibly pruned) SINDy model to ``path``.

    Works for both dynamics models and sparse policies; ``policy_name`` is stored
    when present so the loaded model prints its equations with the right LHS.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = dict(n_out=model.n_out)
    if getattr(model, "policy_name", None) is not None:
        model_cfg["policy_name"] = model.policy_name

    ckpt = {
        "lib_cfg": library_cfg(model.library),
        "model_cfg": model_cfg,
        "active_idx": [list(map(int, s)) for s in model.active_idx],
        "state_dict": model.state_dict(),
    }
    torch.save(ckpt, str(path))
    return path


def load_model(
    path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> SINDyVectorized:
    """Rebuild a :class:`SINDyVectorized` from a checkpoint saved by :func:`save_model`.

    The pruning structure (``active_idx``) is installed *before* the weights are
    loaded so that the ``Xi`` parameter shapes match the checkpoint exactly.
    """
    device = device or torch.device("cpu")
    ckpt = torch.load(str(path), map_location=device)

    lib = CompiledFunctionLibrary(**ckpt["lib_cfg"])
    model = SINDyVectorized(
        library=lib,
        n_out=ckpt["model_cfg"]["n_out"],
        policy_name=ckpt["model_cfg"].get("policy_name"),
        device=device,
    )

    # Install the saved sparsity pattern, resize Xi, then recompile the fast plan.
    model.active_idx = [list(map(int, s)) for s in ckpt["active_idx"]]
    for i, idxs in enumerate(model.active_idx):
        model.Xi[i] = nn.Parameter(
            torch.zeros(len(idxs), 1, device=device), requires_grad=True
        )
    model._plan = model._recompile_fast_path(device=device)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model
