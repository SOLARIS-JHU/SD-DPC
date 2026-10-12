"""Conventional controlled Van der Pol oscillator.

The active case study is the relative-degree-two, single-input formulation

    x0' = x1
    x1' = mu (1 - x0^2) x1 - x0 + u,     mu = 1.

The controlled output is ``x0``: the input first appears in its second derivative. The
sparse policy is pure state feedback and regulates the oscillator to the origin. This
case study supports system identification, policy training, and fixed-policy evaluation;
this repository contains only the one-input formulation.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from neuromancer.psl.signals import step
from neuromancer.psl.base import ODE_NonAutonomous as _ODE
from neuromancer.psl.base import cast_backend

from ..data.generation import get_box_policy_data, get_data
from ..safety.specs import ControlRateConstraint, SafetySpec, box_constraints
from ._losses import tracking_casadi_dpc_objective, tracking_dpc_loss
from .base import System


class VanDerPolControl(_ODE):
    """PSL ground-truth model for the conventional one-input oscillator."""

    @property
    def params(self):
        variables = {
            "x0": self.rng.standard_normal(2),
            "U": 0.5 * np.cos(np.arange(0.0, self.nsim + 1) * 0.02).reshape(-1, 1),
        }
        constants = {"ts": 0.1}
        parameters = {"mu": 1.0}
        return variables, constants, parameters, {}

    @cast_backend
    def equations(self, t, x, u):
        dx0 = x[1]
        dx1 = self.mu * (1 - x[0] ** 2) * x[1] - x[0] + u[0]
        return [dx0, dx1]

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        return step(
            nsim=nsim, d=1, min=-0.5, max=0.5,
            randsteps=int(np.ceil(nsim / 200)), rng=self.rng,
        )


class VanDerPolSystem(System):
    """Relative-degree-two Van der Pol regulation problem with one control input."""

    name = "vanderpol"
    nx, nu, ts = 2, 1, 0.1
    umin, umax = -5.0, 5.0
    xmin, xmax = -5.0, 5.0
    is_discrete = False

    def nominal_params(self) -> Dict:
        return {"mu": 1.0}

    def true_ode(self, x, u, params=None):
        mu = (params or self.nominal_params())["mu"]
        x0, x1 = x[:, [0]], x[:, [1]]
        control = u[:, [0]]
        dx0 = x1
        dx1 = mu * (1 - x0 ** 2) * x1 - x0 + control
        return torch.cat([dx0, dx1], dim=-1)

    def perturbed_sindy_model(self, cfg: Dict):
        mu = float(self.perturbed_params(cfg)["mu"])
        return self.sindy_model_from_terms(
            [
                {"x1": 1.0},
                {"x0": -1.0, "x1": mu, "x0^2*x1": -mu, "u0": 1.0},
            ],
            device=self.device,
        )

    def sindy_library_cfg(self) -> Dict:
        return dict(
            n_features=2, n_control=1, is_policy=False, include_bias=True,
            max_degree=4, add_sqrt=False, add_xu=True, add_u_prod=True,
            add_fourier=False,
        )

    def policy_library_cfg(self) -> Dict:
        return dict(
            n_features=2, n_control=0, is_policy=True, include_bias=False,
            max_degree=4, add_sqrt=False, add_xu=False, add_u_prod=True,
            add_fourier=False, max_freq=2,
        )

    def make_sysid_data(self, cfg: Dict, device):
        return get_data(
            VanDerPolControl(), nsim=cfg.get("nsim", 2000),
            nsteps=cfg.get("nsteps", 2), ts=self.ts,
            bs=cfg.get("bs", 2000), device=device,
            measurement_noise=cfg.get("measurement_noise", 0.0),
        )

    def make_policy_data(self, cfg: Dict, device):
        return get_box_policy_data(
            nsteps=cfg.get("nsteps", 50), n_samples=cfg.get("n_samples", 2000),
            nx=self.nx, xmin=cfg.get("xmin_data", -4.0),
            xmax=cfg.get("xmax_data", 4.0), zero_refs=True,
            same_ref_for_all_states=True, device=device,
            batch_size=cfg.get("batch_size", 200), seed=cfg.get("seed", 0),
        )

    def build_dpc_loss(self, cfg: Dict) -> Tuple[List, List]:
        return tracking_dpc_loss(cfg, self.xmin, self.xmax)

    def casadi_dpc_objective(self, ca, X, U, reference, cfg: Dict):
        return tracking_casadi_dpc_objective(
            ca, X, U, reference, cfg, self.xmin, self.xmax,
            terminal_tol=float(cfg.get("terminal_tol", 1.0e-2)),
        )

    def safety_specs(self, cfg: Dict) -> SafetySpec:
        """Evaluation box/rate specification."""
        w = cfg.get("weights", {})
        bands = cfg.get("bands", {})
        xlo = cfg.get("safe_xmin", self.xmin)
        xhi = cfg.get("safe_xmax", self.xmax)
        cons = box_constraints(
            xlo, xhi, idx=[0, 1], delta=0.0,
            band=bands.get("box", 0.0), weight=w.get("Q_con_x", 1.0),
        )
        du_max = cfg.get("du_max")
        rate = (
            ControlRateConstraint(
                du_max=du_max, band=bands.get("du", 0.0),
                weight=w.get("Q_con_du", 1.0),
            )
            if du_max is not None else None
        )
        return SafetySpec(state_constraints=cons, control_rate=rate)

    def casadi_hooks(self, cfg=None, *, integration_method: str = "rk4"):
        try:
            import casadi as ca
        except Exception:
            return None
        mu = float(self.perturbed_params(cfg or {})["mu"])

        def ode_casadi(x, u):
            return ca.vertcat(
                x[1],
                mu * (1.0 - x[0] ** 2) * x[1] - x[0] + u[0],
            )

        method = str(integration_method).lower()
        if method not in {"euler", "rk4"}:
            raise ValueError("VanDerPol integration_method must be 'euler' or 'rk4'")

        def f_casadi(x, u):
            if method == "euler":
                return x + self.ts * ode_casadi(x, u)
            k1 = ode_casadi(x, u)
            k2 = ode_casadi(x + 0.5 * self.ts * k1, u)
            k3 = ode_casadi(x + 0.5 * self.ts * k2, u)
            k4 = ode_casadi(x + self.ts * k3, u)
            return x + (self.ts / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return {"f_casadi": f_casadi}
