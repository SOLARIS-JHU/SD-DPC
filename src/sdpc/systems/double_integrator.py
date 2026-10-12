"""Double-integrator obstacle-avoidance example (Sec. 5, Obstacle Avoidance).

Discrete plant ``x_{k+1} = A x_k + B u_k`` with a 2-D position state ``x = [x1, x2]`` and
input ``u = [u1, u2]``. The nominal model is the identity/unit-gain map; at deployment the
plant is perturbed (state coupling + halved input gain), which fixed-policy evaluation must
correct while avoiding a keep-out ellipse.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch

from neuromancer.constraint import variable

from ..data.generation import get_data_discrete, get_obstacle_policy_data
from ..safety.specs import (
    ControlRateConstraint,
    SafetySpec,
    box_constraints,
    rotated_ellipse_constraint,
)
from .base import System


class _DiscreteDoubleIntegrator2D:
    """Minimal discrete plant used only for system-ID data generation."""

    def __init__(self, Ad=None, Bd=None, ts=1.0, device=None):
        self.device = device or torch.device("cpu")
        self._nx, self._nu, self.ts = 2, 2, ts
        eye = torch.eye(2, dtype=torch.float32, device=self.device)
        self.Ad = eye.clone() if Ad is None else torch.as_tensor(Ad, dtype=torch.float32, device=self.device)
        self.Bd = eye.clone() if Bd is None else torch.as_tensor(Bd, dtype=torch.float32, device=self.device)
        self._umin = torch.tensor([-1.0, -1.0], device=self.device)
        self._umax = torch.tensor([1.0, 1.0], device=self.device)

    @property
    def nx(self): return self._nx
    @property
    def nu(self): return self._nu

    def sample_x0(self, low=-20.0, high=20.0):
        return (low + (high - low) * torch.rand(self.nx, device=self.device)).float()

    def get_U(self, nsim, **_):
        u = torch.empty(nsim, self.nu, device=self.device)
        idx = 0
        while idx < nsim:
            seg = min(int(torch.randint(1, 21, (1,)).item()), nsim - idx)
            u[idx:idx + seg, :] = 2.0 * torch.rand(self.nu, device=self.device) - 1.0
            idx += seg
        return torch.clamp(u, self._umin, self._umax)

    def step(self, x, u):
        return self.Ad @ x + self.Bd @ u


class DoubleIntegratorSystem(System):
    name = "double_integrator"
    nx, nu, ts = 2, 2, 1.0
    umin, umax = -1.0, 1.0
    xmin, xmax = -5.0, 5.0
    is_discrete = True

    def nominal_params(self) -> Dict:
        return {"A": [[1.0, 0.0], [0.0, 1.0]], "B": [[1.0, 0.0], [0.0, 1.0]]}

    def true_ode(self, x, u, params=None):
        params = params or self.nominal_params()
        A = torch.as_tensor(params["A"], dtype=x.dtype, device=x.device)
        B = torch.as_tensor(params["B"], dtype=x.dtype, device=x.device)
        return x @ A.T + u @ B.T   # discrete map (is_discrete=True)

    def perturbed_params(self, cfg: Dict) -> Dict:
        pert = cfg.get("perturbation", {})
        nominal = self.nominal_params()
        return {
            "A": pert.get("A", nominal["A"]),
            "B": pert.get("B", nominal["B"]),
        }

    def perturbed_sindy_model(self, cfg: Dict):
        params = self.perturbed_params(cfg)
        A = params["A"]
        B = params["B"]
        return self.sindy_model_from_terms(
            [
                {"x0": A[0][0], "x1": A[0][1], "u0": B[0][0], "u1": B[0][1]},
                {"x0": A[1][0], "x1": A[1][1], "u0": B[1][0], "u1": B[1][1]},
            ],
            device=self.device,
        )

    # ---- libraries ------------------------------------------------------ #
    def sindy_library_cfg(self) -> Dict:
        return dict(n_features=2, n_control=2, is_policy=False, include_bias=True,
                    max_degree=2, add_sqrt=True, add_xu=True, add_u_prod=True,
                    add_fourier=True, max_freq=1)

    def policy_library_cfg(self) -> Dict:
        return dict(n_features=2, n_control=2, is_policy=True, include_bias=True,
                    max_degree=2, add_sqrt=False, add_xu=True, add_u_prod=True,
                    add_fourier=True, max_freq=2)

    # ---- data ----------------------------------------------------------- #
    def make_sysid_data(self, cfg: Dict, device):
        plant = _DiscreteDoubleIntegrator2D(ts=self.ts, device=device)
        return get_data_discrete(
            plant, nsim=cfg.get("nsim", 1500), nsteps=cfg.get("nsteps", 2),
            ts=self.ts, bs=cfg.get("bs", 1500), device=device,
            x0_range=tuple(cfg.get("x0_range", (-20.0, 20.0))),
        )

    def make_policy_data(self, cfg: Dict, device):
        ell = cfg["ellipse"]
        return get_obstacle_policy_data(
            nsteps=cfg.get("nsteps", 100), n_samples=cfg.get("n_samples", 3000),
            nx=self.nx, pos_idx=(0, 1), batch_size=cfg.get("batch_size", 400), device=device,
            ellipse_configs={k: ell[k] for k in ("p", "b", "c", "d")},
            init_rect=_square(tuple(cfg["init_center"]), cfg["init_side"]),
            ref_rect=_square(tuple(cfg["ref_center"]), cfg["ref_side"]),
        )

    # ---- DPC training loss (obstacle avoidance) ------------------------- #
    def uses_u_at_ref(self) -> bool:
        return True

    def dpc_loss_spec(self, cfg: Dict) -> Dict:
        weights = cfg.get("weights", {})
        keys = (
            "Q_r", "Q_u", "Q_uf", "Q_dx", "Q_du",
            "Q_con_obs", "Q_con_x", "Q_con_u",
        )
        return {
            "horizon": int(cfg.get("nsteps", 100)),
            "refstep": int(cfg.get("refstep", 1)),
            "weights": {key: float(weights[key]) for key in keys if key in weights},
            "objectives": [
                "terminal_tracking", "control_effort", "state_smoothing",
                "control_smoothing", "terminal_control",
            ],
            "constraints": ["rotated_ellipse", "state_box", "control_box"],
            "policy_only": ["action_at_reference", "l1_sparsity"],
            "mpc_excluded_policy_only": ["action_at_reference", "l1_sparsity"],
        }

    def build_dpc_loss(self, cfg: Dict) -> Tuple[List, List]:
        w = cfg["weights"]
        ell = cfg["ellipse"]
        refstep = cfg.get("refstep", 1)

        x = variable("xn")
        u = variable("u")
        ref = variable("r")
        x1 = variable("xn")[:, :, [0]]
        x2 = variable("xn")[:, :, [1]]

        reference_loss = w["Q_r"] * ((ref[:, -refstep:, :] == x[:, -refstep:, :]) ^ 2)
        reference_loss.name = "reference_loss_position"
        action_loss = w["Q_u"] * ((u == 0.0) ^ 2); action_loss.name = "action_loss"
        state_smoothing = w["Q_dx"] * ((x[:, 1:, :] == x[:, :-1, :]) ^ 2); state_smoothing.name = "state_smoothing"
        control_smoothing = w["Q_du"] * ((u[:, 1:, :] == u[:, :-1, :]) ^ 2); control_smoothing.name = "control_smoothing"
        u_t = variable("u")[:, -1:, :]
        action_final = w["Q_uf"] * ((u_t == 0.0) ^ 2); action_final.name = "action_final"
        u_f = variable("u_f")[:, -1:, :]
        u_at_r = w["Q_uf"] * ((u_f == 0.0) ^ 2); u_at_r.name = "u_at_r"

        objectives = [reference_loss, action_loss, state_smoothing,
                      control_smoothing, action_final, u_at_r]

        # Rotated keep-out ellipse constraint (b*x_rot^2 + y_rot^2 >= (p/2)^2).
        p, b, c, d = ell["p"], ell["b"], ell["c"], ell["d"]
        theta = ell.get("theta", 0.0)
        ct, st = math.cos(theta), math.sin(theta)
        dx, dy = x1 - c, x2 - d
        x_rot = ct * dx + st * dy
        y_rot = -st * dx + ct * dy
        ellipse_expr = b * x_rot ** 2 + y_rot ** 2

        constraints = [
            w["Q_con_obs"] * ((p / 2) ** 2 <= ellipse_expr),
            w["Q_con_x"] * (x > self.xmin),
            w["Q_con_x"] * (x < self.xmax),
            w["Q_con_u"] * (u < self.umax),
            w["Q_con_u"] * (u > self.umin),
        ]
        for con, nm in zip(constraints, ["obstacle_pen", "x_min", "x_max", "u_max", "u_min"]):
            con.name = nm
        return objectives, constraints

    def casadi_dpc_objective(self, ca, X, U, reference, cfg: Dict):
        """CasADi equivalent of the trajectory-dependent SD-DPC obstacle loss."""
        w = cfg["weights"]
        refstep = min(int(cfg.get("refstep", 1)), int(X.shape[1]))
        terminal_state = cfg.get("_terminal_state")
        if terminal_state is not None:
            if refstep != 1:
                raise ValueError("arrival_deadline currently requires refstep=1")
            objective = float(w["Q_r"]) * ca.sumsqr(terminal_state - reference)
        else:
            objective = float(w["Q_r"]) * ca.sumsqr(
                X[:, -refstep:] - ca.repmat(reference, 1, refstep)
            )
        objective += float(w.get("Q_u", 0.0)) * ca.sumsqr(U)
        if X.shape[1] > 1:
            objective += float(w.get("Q_dx", 0.0)) * ca.sumsqr(X[:, 1:] - X[:, :-1])
        if U.shape[1] > 1:
            objective += float(w.get("Q_du", 0.0)) * ca.sumsqr(U[:, 1:] - U[:, :-1])
        if U.shape[1] > 0:
            objective += float(w.get("Q_uf", 0.0)) * ca.sumsqr(U[:, -1])

        if bool(cfg.get("match_dpc_constraints", False)):
            ell = cfg["ellipse"]
            theta = float(ell.get("theta", 0.0))
            ct, st = math.cos(theta), math.sin(theta)
            dx = X[0, :] - float(ell["c"])
            dy = X[1, :] - float(ell["d"])
            x_rot = ct * dx + st * dy
            y_rot = -st * dx + ct * dy
            ellipse_margin = float(ell["b"]) * x_rot ** 2 + y_rot ** 2
            radius_sq = (float(ell["p"]) / 2.0) ** 2
            objective += float(w["Q_con_obs"]) * ca.sumsqr(
                ca.fmax(radius_sq - ellipse_margin, 0.0)
            )
            objective += float(w["Q_con_x"]) * ca.sumsqr(
                ca.fmax(float(self.xmin) - X, 0.0)
            )
            objective += float(w["Q_con_x"]) * ca.sumsqr(
                ca.fmax(X - float(self.xmax), 0.0)
            )
            objective += float(w["Q_con_u"]) * ca.sumsqr(
                ca.fmax(float(self.umin) - U, 0.0)
            )
            objective += float(w["Q_con_u"]) * ca.sumsqr(
                ca.fmax(U - float(self.umax), 0.0)
            )
        return objective

    # ---- safety --------------------------------------------------------- #
    def safety_specs(self, cfg: Dict) -> SafetySpec:
        """Obstacle(s) + box safety spec; override the box with ``cfg['safe_xmin']``/
        ``safe_xmax`` (defaults to the system bounds) for a tighter safety margin than
        the arena limits.
        """
        w = cfg.get("weights", {})
        bands = cfg.get("bands", {})
        ellipses = cfg.get("obstacles")
        if ellipses is None:
            ellipses = [cfg["ellipse"]]
        if not ellipses:
            raise ValueError("DoubleIntegrator safety requires at least one obstacle")

        cons = []
        names = set()
        for i, ell in enumerate(ellipses):
            default_name = "obstacle" if len(ellipses) == 1 else f"obstacle_{i + 1}"
            name = str(ell.get("name", default_name))
            if name in names:
                raise ValueError(f"duplicate obstacle name: {name!r}")
            names.add(name)
            cons.append(rotated_ellipse_constraint(
                p=ell["p"], b=ell["b"], c=ell["c"], d=ell["d"],
                theta=ell.get("theta", 0.0), idx=(0, 1), name=name, delta=0.0,
                band=ell.get("band", bands.get("obstacle", 0.5)),
                weight=ell.get("weight", w.get("Q_con_obs", 1.0)),
            ))
        xlo = cfg.get("safe_xmin", self.xmin)
        xhi = cfg.get("safe_xmax", self.xmax)
        cons += box_constraints(xlo, xhi, idx=[0, 1], delta=0.0,
                                band=bands.get("box", 0.5), weight=w.get("Q_con_x", 1.0))
        du_max = cfg.get("du_max")
        cr = None
        if du_max is not None:
            cr = ControlRateConstraint(du_max=du_max, band=bands.get("du", 0.01),
                                       weight=w.get("Q_con_du", 1.0))
        return SafetySpec(state_constraints=cons, control_rate=cr)

    def obstacle(self, cfg: Dict) -> Optional[Dict]:
        if "ellipse" in cfg:
            return dict(cfg["ellipse"])
        obstacles = cfg.get("obstacles", [])
        return dict(obstacles[0]) if obstacles else None

    # ---- CasADi hooks for the MPC benchmark ----------------------------- #
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
        A = params["A"]
        B = params["B"]

        def f_casadi(x, u):
            Ax = ca.vertcat(A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1])
            Bu = ca.vertcat(B[0][0] * u[0] + B[0][1] * u[1], B[1][0] * u[0] + B[1][1] * u[1])
            return Ax + Bu

        ellipses = (cfg or {}).get("obstacles")
        if ellipses is None:
            ellipse = (cfg or {}).get("ellipse")
            ellipses = [] if ellipse is None else [ellipse]

        def h_casadi(x):
            constraints = []
            for ell in ellipses:
                theta = float(ell.get("theta", 0.0))
                ct, st = math.cos(theta), math.sin(theta)
                dx = x[0] - float(ell["c"])
                dy = x[1] - float(ell["d"])
                xr = ct * dx + st * dy
                yr = -st * dx + ct * dy
                margin = float(ell["b"]) * xr ** 2 + yr ** 2 - (float(ell["p"]) / 2.0) ** 2
                constraints.append((margin, 0.0))
            return constraints

        return {"f_casadi": f_casadi, "h_casadi": h_casadi}


def _square(center: Tuple[float, float], side: float):
    cx, cy = center
    half = side / 2.0
    return ((cx - half, cy - half), (cx + half, cy + half))
