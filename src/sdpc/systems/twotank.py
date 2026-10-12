"""Two-Tank tracking example (Sec. 5, Two-Tank System).

Continuous plant with liquid levels ``x = [x1, x2]`` and inputs ``u = [u1, u2]`` (pump,
valve), matching the PSL ``TwoTank`` convention used for system-ID data:

    x1' = c1 (1 - u2) u1 - c2 sqrt(x1)
    x2' = c1 u2 u1 + c2 sqrt(x1) - c2 sqrt(x2)

with nominal ``c1 = 0.08, c2 = 0.04``. Evaluation checks the levels within the
box ``0 <= x_i <= 1`` under parametric noise on ``c1, c2``.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data.generation import get_box_policy_data, get_data
from ..safety.specs import ControlRateConstraint, SafetySpec, box_constraints
from ._losses import tracking_casadi_dpc_objective, tracking_dpc_loss
from .base import System


class TwoTankSystem(System):
    name = "twotank"
    nx, nu, ts = 2, 2, 1.0
    umin, umax = 0.0, 1.0
    xmin, xmax = 0.0, 1.0
    is_discrete = False

    def nominal_params(self) -> Dict:
        return {"c1": 0.08, "c2": 0.04}

    def true_ode(self, x, u, params=None):
        params = params or self.nominal_params()
        c1, c2 = params["c1"], params["c2"]
        x1, x2 = x[:, [0]], x[:, [1]]
        u1, u2 = u[:, [0]], u[:, [1]]
        sx1 = torch.sqrt(torch.clamp(x1, min=1e-12))
        sx2 = torch.sqrt(torch.clamp(x2, min=1e-12))
        dx1 = c1 * (1 - u2) * u1 - c2 * sx1
        dx2 = c1 * u2 * u1 + c2 * sx1 - c2 * sx2
        return torch.cat([dx1, dx2], dim=-1)

    def perturbed_sindy_model(self, cfg: Dict):
        params = self.perturbed_params(cfg)
        c1, c2 = float(params["c1"]), float(params["c2"])
        return self.sindy_model_from_terms(
            [
                {"u0": c1, "sqrt(x0)": -c2, "u0*u1": -c1},
                {"sqrt(x0)": c2, "sqrt(x1)": -c2, "u0*u1": c1},
            ],
            device=self.device,
        )

    # ---- libraries ------------------------------------------------------ #
    def sindy_library_cfg(self) -> Dict:
        return dict(n_features=2, n_control=2, is_policy=False, include_bias=True,
                    max_degree=2, add_sqrt=True, add_xu=True, add_u_prod=True,
                    add_fourier=False)

    def policy_library_cfg(self) -> Dict:
        return dict(n_features=2, n_control=1, is_policy=True, include_bias=False,
                    max_degree=2, add_sqrt=True, add_xu=True, add_u_prod=True,
                    add_fourier=True, max_freq=2)

    # ---- data ----------------------------------------------------------- #
    def make_sysid_data(self, cfg: Dict, device):
        import neuromancer.psl as psl
        gt = psl.nonautonomous.TwoTank()
        return get_data(gt, nsim=cfg.get("nsim", 2000), nsteps=cfg.get("nsteps", 2),
                        ts=self.ts, bs=cfg.get("bs", 2000), device=device)

    def make_policy_data(self, cfg: Dict, device):
        return get_box_policy_data(
            nsteps=cfg.get("nsteps", 50), n_samples=cfg.get("n_samples", 2000),
            nx=self.nx, xmin=self.xmin, xmax=self.xmax, zero_refs=False,
            same_ref_for_all_states=True, device=device,
            batch_size=cfg.get("batch_size", 200), seed=cfg.get("seed", 5),
        )

    # ---- DPC loss ------------------------------------------------------- #
    def build_dpc_loss(self, cfg: Dict) -> Tuple[List, List]:
        return tracking_dpc_loss(cfg, self.xmin, self.xmax)

    def casadi_dpc_objective(self, ca, X, U, reference, cfg: Dict):
        return tracking_casadi_dpc_objective(
            ca, X, U, reference, cfg, self.xmin, self.xmax,
            terminal_tol=float(cfg.get("terminal_tol", 1.0e-2)),
        )

    # ---- safety --------------------------------------------------------- #
    def safety_specs(self, cfg: Dict) -> SafetySpec:
        """Box safety spec using configurable ``xmin``/``xmax`` evaluation limits.

        The legacy ``safe_xmin``/``safe_xmax`` keys remain supported.
        """
        w = cfg.get("weights", {})
        bands = cfg.get("bands", {})
        xlo = float(cfg.get("xmin", cfg.get("safe_xmin", self.xmin)))
        xhi = float(cfg.get("xmax", cfg.get("safe_xmax", self.xmax)))
        if not self.xmin <= xlo < xhi <= self.xmax:
            raise ValueError(
                f"TwoTank bounds must satisfy {self.xmin} <= xmin < xmax <= {self.xmax}; "
                f"got xmin={xlo}, xmax={xhi}"
            )
        cons = box_constraints(xlo, xhi, idx=[0, 1], delta=0.0,
                               band=bands.get("box", 0.05), weight=w.get("Q_con_x", 1.0))
        du_max = cfg.get("du_max")
        cr = ControlRateConstraint(du_max=du_max, band=bands.get("du", 0.01),
                                   weight=w.get("Q_con_du", 1.0)) if du_max is not None else None
        return SafetySpec(state_constraints=cons, control_rate=cr)

    def casadi_hooks(
        self,
        cfg: Optional[Dict] = None,
        *,
        integration_method: str = "rk4",
    ) -> Optional[Dict[str, Callable]]:
        try:
            import casadi as ca
        except Exception:
            return None
        params = self.perturbed_params(cfg or {})
        c1, c2 = float(params["c1"]), float(params["c2"])

        def ode_casadi(x, u):
            sx1 = ca.sqrt(ca.fmax(x[0], 1e-12))
            sx2 = ca.sqrt(ca.fmax(x[1], 1e-12))
            dx1 = c1 * (1 - u[1]) * u[0] - c2 * sx1
            dx2 = c1 * u[1] * u[0] + c2 * sx1 - c2 * sx2
            return ca.vertcat(dx1, dx2)

        method = integration_method.lower()
        if method not in {"euler", "rk4"}:
            raise ValueError("MPC integration_method must be 'euler' or 'rk4'")

        def f_casadi(x, u):
            if method == "euler":
                return x + self.ts * ode_casadi(x, u)
            k1 = ode_casadi(x, u)
            k2 = ode_casadi(x + 0.5 * self.ts * k1, u)
            k3 = ode_casadi(x + 0.5 * self.ts * k2, u)
            k4 = ode_casadi(x + self.ts * k3, u)
            return x + (self.ts / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        def mpc_u_guess(_x0, reference):
            # For x1 = x2 = r, the nominal steady input is
            # u1 = (c2 / c1) sqrt(r), u2 = 0.
            target = np.clip(np.asarray(reference[:-1, 0], dtype=float), 0.0, None)
            u1 = np.clip((c2 / c1) * np.sqrt(target), self.umin, self.umax)
            return np.column_stack([u1, np.zeros_like(u1)])

        return {
            "f_casadi": f_casadi,
            "mpc_u_guess": mpc_u_guess,
        }
