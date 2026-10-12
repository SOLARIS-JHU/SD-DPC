"""Reference-signal builders for fixed-controller evaluation."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

__all__ = [
    "make_constant_reference",
    "make_piecewise_reference",
    "make_equal_piecewise_reference",
    "make_signal_reference",
    "build_reference",
]

Level = Union[float, Sequence[float]]


def make_constant_reference(nx: int, nsteps: int, level: Level, device=None) -> torch.Tensor:
    """Constant reference held for the whole horizon: ``(1, nsteps+1, nx)``."""
    device = device or torch.device("cpu")
    lvl = torch.as_tensor(level, dtype=torch.float32, device=device)
    if lvl.dim() == 0:
        lvl = lvl.expand(nx)
    return lvl.reshape(1, 1, nx).expand(1, nsteps + 1, nx).contiguous()


def make_piecewise_reference(
    nx: int,
    segments: Sequence[Tuple[Level, int]],
    device=None,
) -> torch.Tensor:
    """Piecewise-constant reference from ``[(level, hold_steps), ...]`` segments.

    Each ``level`` is a scalar (broadcast to all ``nx`` channels) or a length-``nx``
    sequence; the reference holds that level for ``hold_steps`` steps before switching to
    the next segment. Returns ``(1, sum(hold_steps), nx)`` — pick ``hold_steps`` so the
    total matches the run's ``nsteps + 1``.
    """
    device = device or torch.device("cpu")
    chunks = []
    for level, hold in segments:
        lvl = torch.as_tensor(level, dtype=torch.float32, device=device)
        if lvl.dim() == 0:
            lvl = lvl.expand(nx)
        chunks.append(lvl.reshape(1, 1, nx).expand(1, int(hold), nx))
    return torch.cat(chunks, dim=1).contiguous()


def make_equal_piecewise_reference(
    nx: int,
    nsteps: int,
    levels: Sequence[Level],
    device=None,
) -> torch.Tensor:
    """Hold each level for an equal share of ``nsteps`` control intervals.

    The returned tensor contains ``nsteps + 1`` samples. The extra sample at ``t=0``
    uses the first level; the remaining ``nsteps`` samples are divided across all levels.
    Any remainder is assigned one step at a time from the first level onward.
    """
    nsteps = int(nsteps)
    levels = list(levels)
    if nsteps < 1:
        raise ValueError("nsteps must be positive")
    if not levels:
        raise ValueError("equal reference requires at least one level")
    if len(levels) > nsteps:
        raise ValueError("number of reference levels cannot exceed nsteps")

    base, remainder = divmod(nsteps, len(levels))
    holds = [base + (i < remainder) for i in range(len(levels))]
    holds[0] += 1  # Include the initial t=0 reference sample.
    return make_piecewise_reference(nx, list(zip(levels, holds)), device=device)


def make_signal_reference(
    nx: int,
    nsteps: int,
    kind: str = "step",
    *,
    seed: Optional[int] = None,
    device=None,
    **signal_kwargs,
) -> torch.Tensor:
    """Reference built from a ``neuromancer.psl.signals`` generator.

    ``kind`` is one of ``step`` (piecewise-constant random levels, the default),
    ``sines``, ``periodic``, ``noise``, or ``walk``. ``signal_kwargs`` (e.g. ``min``,
    ``max``, ``randsteps``) are forwarded to the generator. Returns ``(1, nsteps+1, nx)``.
    """
    from neuromancer.psl import signals

    fn = getattr(signals, kind)
    rng = np.random.default_rng(seed)
    sig = fn(nsim=nsteps + 1, d=nx, rng=rng, **signal_kwargs)
    return torch.as_tensor(np.asarray(sig), dtype=torch.float32, device=device).unsqueeze(0)


def build_reference(
    nx: int,
    nsteps: int,
    cfg: Optional[Dict],
    *,
    seed: Optional[int] = None,
    device=None,
    bounds: Optional[Tuple[float, float]] = None,
    clearance: float = 0.0,
) -> torch.Tensor:
    """Dispatch on ``cfg['kind']``: ``constant``, ``manual`` (piecewise segments),
    ``equal`` (equally held levels), or a ``neuromancer.psl.signals`` name. When
    ``relative_to_bounds`` is true, configured values are fractions of ``bounds``.
    If bounds are provided, every resolved reference is checked against ``clearance``.

    Example configs::

        reference: {kind: constant, level: [0.6, 0.6]}
        reference: {kind: manual, segments: [[[0.3, 0.3], 100], [[0.7, 0.7], 100]]}
        reference: {kind: equal, relative_to_bounds: true,
                    levels: [[0.3, 0.3], [0.8, 0.6], [0.5, 0.5]]}
        reference: {kind: step, min: 0.2, max: 0.9, randsteps: 6}
    """
    if not cfg:
        raise ValueError("reference config must define at least 'kind'")
    kind = cfg.get("kind", "constant")
    if kind == "constant":
        reference = make_constant_reference(nx, nsteps, cfg["level"], device=device)
    elif kind == "manual":
        segments = [tuple(s) for s in cfg["segments"]]
        reference = make_piecewise_reference(nx, segments, device=device)
    elif kind == "equal":
        reference = make_equal_piecewise_reference(nx, nsteps, cfg["levels"], device=device)
    else:
        signal_kwargs = {
            k: v for k, v in cfg.items()
            if k not in {"kind", "relative_to_bounds"}
        }
        reference = make_signal_reference(
            nx, nsteps, kind=kind, seed=seed, device=device, **signal_kwargs
        )

    relative = bool(cfg.get("relative_to_bounds", False))
    if relative:
        if bounds is None:
            raise ValueError("relative_to_bounds references require xmin/xmax bounds")
        tol = 1.0e-7
        if float(reference.min()) < -tol or float(reference.max()) > 1.0 + tol:
            raise ValueError("relative reference levels must lie in [0, 1]")
        xmin, xmax = map(float, bounds)
        reference = xmin + (xmax - xmin) * reference

    if bounds is not None:
        xmin, xmax = map(float, bounds)
        clearance = float(clearance)
        if xmin >= xmax:
            raise ValueError("reference bounds require xmin < xmax")
        if clearance < 0.0:
            raise ValueError("reference clearance must be nonnegative")
        span = xmax - xmin
        overlap_tol = 1.0e-12 * max(1.0, span)
        if 2.0 * clearance >= span - overlap_tol:
            raise ValueError(
                f"box clearance {clearance} overlaps across bounds [{xmin}, {xmax}]"
            )
        allowed_min, allowed_max = xmin + clearance, xmax - clearance
        tol = 1.0e-7 * max(1.0, xmax - xmin)
        actual_min, actual_max = float(reference.min()), float(reference.max())
        if actual_min < allowed_min - tol or actual_max > allowed_max + tol:
            raise ValueError(
                f"references [{actual_min:.6g}, {actual_max:.6g}] conflict with box "
                f"clearance {clearance}; choose references inside "
                f"[{allowed_min:.6g}, {allowed_max:.6g}]"
            )
    return reference
