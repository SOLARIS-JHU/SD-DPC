"""The :class:`System` interface — the single extension point for new examples.

A system bundles everything example-specific behind a uniform API so the scripts,
trainers and evaluators stay generic:

* dimensions and bounds (``nx, nu, ts, umin, umax, xmin, xmax``);
* the true continuous ODE in torch (``true_ode``) and its nominal parameters;
* library configs for the SINDy dynamics model and the sparse policy;
* data generation for system-ID and for policy training;
* the DPC training loss (objectives + constraints), built on Neuromancer variables;
* the :class:`~sdpc.safety.SafetySpec` used by controller evaluation;
* plants: ``discrete_step`` (nominal model step) and ``perturbed_plant`` (deployment).

To add a system, subclass :class:`System`, implement the abstract members, and register
it in :mod:`sdpc.registry`.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch

from ..safety.specs import SafetySpec


def rk4_step(ode: Callable, x: torch.Tensor, u: torch.Tensor, ts: float) -> torch.Tensor:
    """One classic RK4 step of ``x' = ode(x, u)`` over ``ts`` (continuous-time systems)."""
    k1 = ode(x, u)
    k2 = ode(x + 0.5 * ts * k1, u)
    k3 = ode(x + 0.5 * ts * k2, u)
    k4 = ode(x + ts * k3, u)
    return x + (ts / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class System:
    """Base class for all examples. Subclasses set the attributes / methods below."""

    name: str = "system"
    nx: int = 0
    nu: int = 0
    ts: float = 1.0
    umin: float = -1.0
    umax: float = 1.0
    xmin: float = -1.0
    xmax: float = 1.0
    is_discrete: bool = False   # True => discrete_step uses the model directly (no RK4)

    def __init__(self, device: Optional[torch.device] = None, **overrides):
        self.device = device or torch.device("cpu")
        for k, v in overrides.items():
            setattr(self, k, v)

    # ---- dynamics ------------------------------------------------------- #
    def true_ode(self, x: torch.Tensor, u: torch.Tensor, params: Optional[Dict] = None) -> torch.Tensor:
        """Continuous-time vector field ``x' = f(x, u; params)`` in torch (batched)."""
        raise NotImplementedError

    def nominal_params(self) -> Dict[str, float]:
        return {}

    def true_step(self, x, u, params=None) -> torch.Tensor:
        """One discrete step of the *true* system (RK4 unless the system is discrete)."""
        if self.is_discrete:
            return self.true_ode(x, u, params)
        return rk4_step(lambda xx, uu: self.true_ode(xx, uu, params), x, u, self.ts)

    def discrete_step(self, dynamics_model) -> Callable:
        """Return the nominal one-step map ``f(x, u) -> x_next`` from a SINDy model.

        Discrete systems use the model directly; continuous ones integrate it with RK4.
        Used by policy training and fixed-controller evaluation.
        """
        if self.is_discrete:
            return lambda x, u: dynamics_model(x, u)
        return lambda x, u: rk4_step(dynamics_model.ode_equations, x, u, self.ts)

    def perturbed_plant(self, cfg: Dict) -> Callable:
        """Return the deployment plant ``f(x, u) -> x_next`` with perturbed parameters."""
        params = self.perturbed_params(cfg)
        return lambda x, u: self.true_step(x, u, params)

    def perturbed_params(self, cfg: Dict) -> Dict[str, float]:
        """Nominal params updated by ``cfg['perturbation']`` (system-specific)."""
        params = dict(self.nominal_params())
        params.update(cfg.get("perturbation", {}))
        return params

    def sindy_model_from_terms(self, terms: List[Dict[str, float]], device=None):
        """Build a sparse ``SINDyVectorized`` model from explicit row-term coefficients."""
        import torch.nn as nn

        from ..sindy import CompiledFunctionLibrary, SINDyVectorized

        device = device or self.device
        lib = CompiledFunctionLibrary(**self.sindy_library_cfg())
        model = SINDyVectorized(lib, n_out=self.nx, device=device)
        name_to_idx = {name: i for i, name in enumerate(lib.function_names)}

        active_idx = []
        for row in terms:
            missing = [name for name in row if name not in name_to_idx]
            if missing:
                raise KeyError(f"terms not in {self.name} SINDy library: {missing}")
            idxs = [name_to_idx[name] for name, coef in row.items() if float(coef) != 0.0]
            if not idxs:
                idxs = [0]
            active_idx.append(idxs)

        model.active_idx = active_idx
        for i, row_idxs in enumerate(active_idx):
            coefs = [float(terms[i].get(lib.function_names[j], 0.0)) for j in row_idxs]
            model.Xi[i] = nn.Parameter(
                torch.tensor(coefs, dtype=torch.float32, device=device).view(-1, 1),
                requires_grad=False,
            )
        model._plan = model._recompile_fast_path(device=device)
        return model

    def perturbed_sindy_model(self, cfg: Dict):
        """Exact SINDy-style model of the configured perturbed plant/vector field."""
        raise NotImplementedError(f"{self.name} does not define an exact perturbed SINDy model")

    # ---- SINDy / policy library configs --------------------------------- #
    def sindy_library_cfg(self) -> Dict:
        raise NotImplementedError

    def policy_library_cfg(self) -> Dict:
        raise NotImplementedError

    # ---- data ----------------------------------------------------------- #
    def make_sysid_data(self, cfg: Dict, device):
        raise NotImplementedError

    def make_policy_data(self, cfg: Dict, device):
        raise NotImplementedError

    # ---- policy DPC loss ------------------------------------------------ #
    def build_dpc_loss(self, cfg: Dict) -> Tuple[List, List]:
        """Return ``(objectives, constraints)`` as Neuromancer constraint objects.

        Uses ``neuromancer.constraint.variable`` handles for ``xn``, ``u``, ``r`` (and,
        where relevant, ``u_f`` = policy action evaluated at the reference).
        """
        raise NotImplementedError

    def uses_u_at_ref(self) -> bool:
        """Whether the training graph needs the ``u_f = pi(r, r)`` node."""
        return False

    def dpc_loss_spec(self, cfg: Dict) -> Dict:
        """Serializable description of the SD-DPC loss used for matched baselines."""
        weights = cfg.get("weights", {})
        keys = ("Q_r", "Q_u", "Q_dx", "Q_du", "Q_con")
        return {
            "horizon": int(cfg.get("nsteps", 100)),
            "weights": {key: float(weights[key]) for key in keys if key in weights},
            "objectives": ["tracking", "control_effort", "state_smoothing", "control_smoothing"],
            "constraints": ["state_box", "terminal_reference_box"],
            "policy_only": ["l1_sparsity"],
        }

    def casadi_dpc_objective(self, ca, X, U, reference, cfg: Dict):
        """Return the CasADi form of the non-sparsity SD-DPC loss."""
        raise NotImplementedError(f"{self.name} does not define a CasADi DPC objective")

    # ---- safety --------------------------------------------------------- #
    def safety_specs(self, cfg: Dict) -> SafetySpec:
        raise NotImplementedError

    # ---- optional CasADi hooks for the MPC benchmark -------------------- #
    def casadi_hooks(
        self,
        cfg: Optional[Dict] = None,
        *,
        integration_method: str = "rk4",
    ) -> Optional[Dict[str, Callable]]:
        """Return exact configured CasADi dynamics for the optional MPC baseline."""
        return None

    # ---- plotting hook -------------------------------------------------- #
    def obstacle(self, cfg: Dict) -> Optional[Dict]:
        """Ellipse params for plotting, or None for box-only systems."""
        return None
