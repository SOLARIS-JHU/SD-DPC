"""State and control constraint specifications for evaluation and MPC.

Conventions
-----------
* Every constraint exposes a *margin* function ``h(x)`` where the state is safe iff
  ``h(x) >= delta``. ``h`` is written so that its magnitude is a meaningful distance to
  the boundary (used for the "minimum safety margin" metric).
* The obstacle constraint uses the same rotated-ellipse form as the DoubleIntegrator
  notebooks: ``h = b*x_rot^2 + y_rot^2 - (p/2)^2``.
* The control-rate constraint is ``h_du = du_max^2 - ||u_k - u_{k-1}||^2``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "StateConstraint",
    "ControlRateConstraint",
    "SafetySpec",
    "box_constraints",
    "rotated_ellipse_constraint",
]


@dataclass
class StateConstraint:
    """A scalar state safety function; ``x`` is safe iff ``fn(x) >= delta``.

    ``fn`` maps a state tensor of shape ``(..., nx)`` to a margin tensor of shape
    ``(...)``. ``band`` is an optional extra clearance beyond the hard threshold
    ``delta``; ``weight`` scales a soft constraint penalty.
    """

    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    grad_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    meta: Dict = field(default_factory=dict)
    delta: float = 0.0
    band: float = 0.0
    weight: float = 1.0

    def margin(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_fn is None:
            raise ValueError(
                f"state constraint {self.name!r} has no analytic grad_fn; "
                "symbolic safety gradients require one"
            )
        return self.grad_fn(x)


@dataclass
class ControlRateConstraint:
    """Limit on the per-step control increment: ``||u_k - u_{k-1}|| <= du_max``.

    Encoded as the margin ``h_du = du_max^2 - ||u_k - u_{k-1}||^2`` (safe iff ``>= delta``).
    """

    du_max: float
    name: str = "du"
    delta: float = 0.0
    band: float = 0.0
    weight: float = 1.0

    def margin(self, du_sq_norm: torch.Tensor) -> torch.Tensor:
        return self.du_max ** 2 - du_sq_norm


@dataclass
class SafetySpec:
    """A collection of state constraints plus an optional control-rate constraint."""

    state_constraints: List[StateConstraint] = field(default_factory=list)
    control_rate: Optional[ControlRateConstraint] = None

    def margins(self, x_traj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Evaluate every *state* constraint over a trajectory ``x_traj`` (B, T, nx)."""
        return {c.name: c.margin(x_traj) for c in self.state_constraints}

    def min_margins(self, x_traj: torch.Tensor) -> Dict[str, float]:
        return {k: float(v.min().item()) for k, v in self.margins(x_traj).items()}

    def any_violation(self, x_traj: torch.Tensor) -> bool:
        """True if any state constraint is violated (``h < delta``) anywhere."""
        for c in self.state_constraints:
            if bool((c.margin(x_traj) < c.delta).any().item()):
                return True
        return False


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def box_constraints(
    xmin: float,
    xmax: float,
    idx: Sequence[int],
    *,
    delta: float = 0.0,
    band: float = 0.0,
    weight: float = 1.0,
) -> List[StateConstraint]:
    """Lower/upper box bounds ``xmin <= x_i <= xmax`` for each state index in ``idx``."""
    cons: List[StateConstraint] = []
    for i in idx:
        cons.append(
            StateConstraint(
                name=f"x{i}_min",
                fn=(lambda x, i=i: x[..., i] - xmin),
                grad_fn=(
                    lambda x, i=i: torch.nn.functional.one_hot(
                        torch.as_tensor(i, device=x.device), num_classes=x.shape[-1]
                    ).to(dtype=x.dtype).expand(*x.shape[:-1], x.shape[-1])
                ),
                meta={"kind": "box_min", "idx": int(i), "bound": float(xmin)},
                delta=delta,
                band=band,
                weight=weight,
            )
        )
        cons.append(
            StateConstraint(
                name=f"x{i}_max",
                fn=(lambda x, i=i: xmax - x[..., i]),
                grad_fn=(
                    lambda x, i=i: -torch.nn.functional.one_hot(
                        torch.as_tensor(i, device=x.device), num_classes=x.shape[-1]
                    ).to(dtype=x.dtype).expand(*x.shape[:-1], x.shape[-1])
                ),
                meta={"kind": "box_max", "idx": int(i), "bound": float(xmax)},
                delta=delta,
                band=band,
                weight=weight,
            )
        )
    return cons


def rotated_ellipse_constraint(
    *,
    p: float,
    b: float,
    c: float,
    d: float,
    theta: float = 0.0,
    idx: Tuple[int, int] = (0, 1),
    name: str = "obstacle",
    delta: float = 0.0,
    band: float = 0.0,
    weight: float = 1.0,
) -> StateConstraint:
    """Keep-out constraint for a rotated ellipse (safe = *outside*).

    ``h(x) = b*x_rot^2 + y_rot^2 - (p/2)^2``, matching the DoubleIntegrator obstacle.
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    boundary = (p / 2.0) ** 2
    ix, iy = idx

    def _h(x: torch.Tensor) -> torch.Tensor:
        dx = x[..., ix] - c
        dy = x[..., iy] - d
        x_rot = ct * dx + st * dy
        y_rot = -st * dx + ct * dy
        return b * x_rot ** 2 + y_rot ** 2 - boundary

    def _grad(x: torch.Tensor) -> torch.Tensor:
        dx = x[..., ix] - c
        dy = x[..., iy] - d
        x_rot = ct * dx + st * dy
        y_rot = -st * dx + ct * dy
        gx = 2.0 * b * x_rot * ct - 2.0 * y_rot * st
        gy = 2.0 * b * x_rot * st + 2.0 * y_rot * ct
        out = torch.zeros_like(x)
        out[..., ix] = gx
        out[..., iy] = gy
        return out

    return StateConstraint(
        name=name,
        fn=_h,
        grad_fn=_grad,
        meta={
            "kind": "rotated_ellipse",
            "idx": tuple(int(i) for i in idx),
            "p": float(p),
            "b": float(b),
            "c": float(c),
            "d": float(d),
            "theta": float(theta),
            "cos_theta": float(ct),
            "sin_theta": float(st),
            "boundary": float(boundary),
        },
        delta=delta, band=band, weight=weight,
    )
