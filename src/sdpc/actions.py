"""Shared bounded-action semantics for training and evaluation."""
from __future__ import annotations

import torch

ACTION_GRADIENT_MODES = {"exact", "straight_through", "leaky_straight_through"}

__all__ = ["ACTION_GRADIENT_MODES", "action_gradient_mask", "bounded_action"]


def action_gradient_mask(
    scaled_action: torch.Tensor,
    umin,
    umax,
    *,
    gradient_mode: str = "exact",
    gradient_band: float = 0.1,
    gradient_leak: float = 0.05,
) -> torch.Tensor:
    """Return the clamp surrogate derivative evaluated at a scaled raw action."""
    if gradient_mode not in ACTION_GRADIENT_MODES:
        raise ValueError(
            f"gradient_mode must be one of {sorted(ACTION_GRADIENT_MODES)}, "
            f"got {gradient_mode!r}"
        )
    if gradient_band < 0.0:
        raise ValueError("gradient_band must be nonnegative")
    if not 0.0 <= gradient_leak <= 1.0:
        raise ValueError("gradient_leak must lie in [0, 1]")

    lo = umin if (umin is None or torch.is_tensor(umin)) else float(umin)
    hi = umax if (umax is None or torch.is_tensor(umax)) else float(umax)
    if torch.is_tensor(lo):
        lo = lo.to(scaled_action)
    if torch.is_tensor(hi):
        hi = hi.to(scaled_action)

    mask = torch.ones_like(scaled_action)
    if gradient_mode == "straight_through":
        return mask
    if gradient_mode == "exact":
        if lo is not None:
            mask = mask * (scaled_action > lo).to(mask.dtype)
        if hi is not None:
            mask = mask * (scaled_action < hi).to(mask.dtype)
        return mask

    near_bounds = torch.ones_like(scaled_action, dtype=torch.bool)
    if lo is not None:
        near_bounds &= scaled_action >= lo - float(gradient_band)
    if hi is not None:
        near_bounds &= scaled_action <= hi + float(gradient_band)
    return torch.where(
        near_bounds,
        mask,
        torch.full_like(mask, float(gradient_leak)),
    )


def bounded_action(
    raw_action: torch.Tensor,
    umin,
    umax,
    action_scale: float = 1.0,
    *,
    gradient_mode: str = "exact",
    gradient_band: float = 0.1,
    gradient_leak: float = 0.05,
) -> torch.Tensor:
    """Scale and hard-clamp an action with a selectable surrogate derivative.

    Every mode has the same hard-clamped forward value. ``exact`` uses the clamp's
    derivative, ``straight_through`` uses derivative one everywhere, and
    ``leaky_straight_through`` uses derivative one within ``gradient_band`` of the
    bounds and ``gradient_leak`` farther outside them. The band is measured after
    ``action_scale`` is applied.
    """
    if gradient_mode not in ACTION_GRADIENT_MODES:
        raise ValueError(
            f"gradient_mode must be one of {sorted(ACTION_GRADIENT_MODES)}, got {gradient_mode!r}"
        )
    lo = umin if (umin is None or torch.is_tensor(umin)) else float(umin)
    hi = umax if (umax is None or torch.is_tensor(umax)) else float(umax)
    if torch.is_tensor(lo):
        lo = lo.to(raw_action)
    if torch.is_tensor(hi):
        hi = hi.to(raw_action)

    scaled = float(action_scale) * raw_action
    if lo is None and hi is None:
        return scaled
    clamped = torch.clamp(scaled, lo, hi)
    if gradient_mode == "straight_through" and torch.is_grad_enabled():
        return scaled + (clamped - scaled).detach()
    if gradient_mode == "leaky_straight_through" and torch.is_grad_enabled():
        slope = action_gradient_mask(
            scaled, lo, hi,
            gradient_mode=gradient_mode,
            gradient_band=gradient_band,
            gradient_leak=gradient_leak,
        )
        return clamped.detach() + slope.detach() * (scaled - scaled.detach())
    return clamped
