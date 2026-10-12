"""Shared DPC loss builders (Neuromancer constraint objects).

The reference-tracking loss (Eq. 24 for Two-Tank, and its regulation form for Van der Pol
where ``r = 0``) is identical in structure across the box-constrained examples, so it lives
here and is reused by both systems. The obstacle-avoidance loss is specific to the
DoubleIntegrator and stays in that system module.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from neuromancer.constraint import variable

__all__ = ["tracking_dpc_loss", "tracking_casadi_dpc_objective"]


def tracking_dpc_loss(
    cfg: Dict,
    xmin: float,
    xmax: float,
    terminal_tol: float = 1e-2,
) -> Tuple[List, List]:
    """Reference-tracking objectives + box/terminal constraints (Sec. 5, Eq. 24).

    ``cfg['weights']`` keys: ``Q_r`` (tracking), ``Q_u`` (effort), ``Q_dx`` (state
    smoothing), ``Q_du`` (control smoothing), ``Q_con`` (constraint penalty).
    """
    w = cfg["weights"]
    x = variable("xn")
    ref = variable("r")
    u = variable("u")

    reg = w["Q_r"] * ((x == ref) ^ 2); reg.name = "reference_loss_position"
    action_loss = w.get("Q_u", 0.0) * ((u == 0.0) ^ 2); action_loss.name = "action_loss"
    state_smoothing = w.get("Q_dx", 0.0) * ((x[:, 1:, :] == x[:, :-1, :]) ^ 2)
    state_smoothing.name = "state_smoothing"
    control_smoothing = w.get("Q_du", 0.0) * ((u[:, 1:, :] == u[:, :-1, :]) ^ 2)
    control_smoothing.name = "control_smoothing"
    objectives = [reg, action_loss, state_smoothing, control_smoothing]

    c = w["Q_con"]
    constraints = [
        c * (x > xmin),
        c * (x < xmax),
        c * (x[:, [-1], :] > ref - terminal_tol),
        c * (x[:, [-1], :] < ref + terminal_tol),
    ]
    for con, nm in zip(constraints, ["x_min", "x_max", "y_N_min", "y_N_max"]):
        con.name = nm
    return objectives, constraints


def tracking_casadi_dpc_objective(
    ca,
    X,
    U,
    reference,
    cfg: Dict,
    xmin: float,
    xmax: float,
    terminal_tol: float = 1e-2,
):
    """CasADi equivalent of :func:`tracking_dpc_loss` for matched MPC."""
    w = cfg["weights"]
    objective = float(w["Q_r"]) * ca.sumsqr(X - ca.repmat(reference, 1, X.shape[1]))
    objective += float(w.get("Q_u", 0.0)) * ca.sumsqr(U)
    if X.shape[1] > 1:
        objective += float(w.get("Q_dx", 0.0)) * ca.sumsqr(X[:, 1:] - X[:, :-1])
    if U.shape[1] > 1:
        objective += float(w.get("Q_du", 0.0)) * ca.sumsqr(U[:, 1:] - U[:, :-1])

    if bool(cfg.get("match_dpc_constraints", False)):
        qcon = float(w["Q_con"])
        objective += qcon * ca.sumsqr(ca.fmax(float(xmin) - X, 0.0))
        objective += qcon * ca.sumsqr(ca.fmax(X - float(xmax), 0.0))
        terminal = X[:, -1]
        objective += qcon * ca.sumsqr(
            ca.fmax(reference - float(terminal_tol) - terminal, 0.0)
        )
        objective += qcon * ca.sumsqr(
            ca.fmax(terminal - reference - float(terminal_tol), 0.0)
        )
    return objective
