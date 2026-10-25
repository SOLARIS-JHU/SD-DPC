"""Distill a trained NN-DPC policy onto an existing sparse policy support."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from ..eval.evaluate import sample_scenario
from ..eval.metrics import compute_metrics, policy_memory_footprint
from ..eval.rollout import rollout_closed_loop
from ..eval.twotank import make_twotank_scenarios

__all__ = [
    "DistilledPolicy",
    "distill_nn_policy",
    "evaluate_distilled_policy",
    "load_distilled_policy",
    "make_distillation_scenarios",
    "save_distilled_policy",
    "validate_reference_results",
]


class DistilledPolicy(torch.nn.Module):
    """Fixed sparse policy with one coefficient vector per control output.

    The support comes from a trained SD-DPC policy. Only its coefficients are refit
    to imitate the NN-DPC teacher, so the distilled controller has the same terms and
    indexed memory footprint as the sparse controller.
    """

    def __init__(
        self,
        library,
        active_idx: Sequence[Sequence[int]],
        coefficients: Sequence[Sequence[float]],
        *,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.library = library
        self.active_idx = [list(map(int, support)) for support in active_idx]
        if len(self.active_idx) != len(coefficients):
            raise ValueError("active_idx and coefficients must have one entry per output")
        self.Xi = torch.nn.ParameterList()
        for support, values in zip(self.active_idx, coefficients):
            tensor = torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1, 1)
            if tensor.shape[0] != len(support):
                raise ValueError("each coefficient vector must match its support")
            self.Xi.append(torch.nn.Parameter(tensor, requires_grad=False))

        union = sorted({index for support in self.active_idx for index in support})
        self._plan = self.library.compile(union, device=torch.device(device))
        self._local_idx: List[torch.Tensor] = []
        for support in self.active_idx:
            global_idx = torch.as_tensor(support, dtype=torch.long, device=device)
            self._local_idx.append(self._plan.global2local[global_idx])

    def forward(self, x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        theta = self.library.evaluate_plan(x, reference, self._plan)
        outputs = [
            theta.index_select(1, local) @ coefficient
            for local, coefficient in zip(self._local_idx, self.Xi)
        ]
        return torch.cat(outputs, dim=1)

    @property
    def dense_coefficients(self) -> torch.Tensor:
        dense = torch.zeros(
            self.library.shape[0], len(self.Xi),
            dtype=self.Xi[0].dtype, device=self.Xi[0].device,
        )
        for output, (support, coefficient) in enumerate(zip(self.active_idx, self.Xi)):
            dense[support, output] = coefficient[:, 0]
        return dense


def make_distillation_scenarios(
    case: str,
    system,
    cfg: Dict,
    *,
    count: int,
    seed: int,
    horizon: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Create the deterministic scenario batch used for fitting or validation."""
    settings = copy.deepcopy(cfg)
    settings.update(seed=int(seed), nsteps=int(horizon), eval_nsteps=int(horizon))
    if case == "TwoTank":
        settings["n_trajectories"] = int(count)
        return make_twotank_scenarios(system, settings, device)
    scenarios = [
        sample_scenario(system, settings, int(seed) + index, device)
        for index in range(int(count))
    ]
    return {
        "xn": torch.cat([scenario["xn"] for scenario in scenarios], dim=0),
        "r": torch.cat([scenario["r"] for scenario in scenarios], dim=0),
    }


def _rollout(policy, system, scenarios):
    plant = lambda x, u: system.true_step(x, u, system.nominal_params())
    return rollout_closed_loop(
        policy, plant, scenarios,
        umin=system.umin, umax=system.umax, action_scale=1.0,
    )


def _library_matrix(source_policy, states, references) -> np.ndarray:
    plan = source_policy.library.compile(device=states.device)
    with torch.no_grad():
        return source_policy.library.evaluate_plan(states, references, plan).cpu().numpy().astype(np.float64)


def distill_nn_policy(system, sparse_policy, teacher, scenarios) -> tuple[DistilledPolicy, Dict]:
    """Fit unregularized least squares coefficients on the sparse policy support."""
    teacher.eval()
    teacher_rollout = _rollout(teacher, system, scenarios)
    states = teacher_rollout["x_traj"][:, :-1].reshape(-1, system.nx)
    references = scenarios["r"][:, :-1].reshape(-1, system.nx)
    actions = teacher_rollout["u_traj"].reshape(-1, system.nu).cpu().numpy()
    theta = _library_matrix(sparse_policy, states, references)

    fitted = []
    residuals = []
    for output, active in enumerate(sparse_policy.active_idx):
        support = list(active)
        estimate = np.linalg.lstsq(theta[:, support], actions[:, output], rcond=None)[0]
        fitted.append(estimate.tolist())
        residuals.append(float(np.mean((theta[:, support] @ estimate - actions[:, output]) ** 2)))

    policy = DistilledPolicy(
        sparse_policy.library, sparse_policy.active_idx, fitted,
        device=states.device,
    ).eval()
    report = {
        "fit": "ordinary_least_squares",
        "training_samples": int(states.shape[0]),
        "support_indices": [list(map(int, support)) for support in sparse_policy.active_idx],
        "terms_per_output": [len(support) for support in sparse_policy.active_idx],
        "teacher_action_mse_per_output": residuals,
    }
    return policy, report


def _task_loss(case: str, x: torch.Tensor, reference: torch.Tensor) -> float:
    horizon = min(x.shape[1], reference.shape[1])
    if case == "TwoTank":
        segments = [
            ((x[:, max(stop - 100, 0):stop] - reference[:, max(stop - 100, 0):stop]) ** 2).mean()
            for stop in range(250, horizon + 1, 250)
        ]
        return float(torch.stack(segments).mean())
    if case == "VanDerPol":
        start = max(0, horizon - 10)
        return float(((x[:, start:horizon] - reference[:, start:horizon]) ** 2).mean())
    return float(((x[:, horizon - 1] - reference[:, horizon - 1]) ** 2).mean())


def evaluate_distilled_policy(case: str, system, policy, scenarios, cfg: Dict) -> Dict:
    """Evaluate one distilled policy with the metrics used in the comparison table."""
    result = _rollout(policy, system, scenarios)
    spec = system.safety_specs(cfg)
    spec.control_rate = None
    task_loss, effort, violation, max_depth = [], [], [], []
    for index in range(result["x_traj"].shape[0]):
        x = result["x_traj"][index:index + 1]
        u = result["u_traj"][index:index + 1]
        reference = scenarios["r"][index:index + 1]
        metrics = compute_metrics(x, u, reference, spec=spec, policy=policy)
        shortfalls = [
            torch.clamp(constraint.delta - 1.0e-7 - constraint.margin(x), min=0.0)
            for constraint in spec.state_constraints
        ]
        task_loss.append(_task_loss(case, x, reference))
        effort.append(metrics["total_control_action_l1"])
        violation.append(int(metrics["num_state_violations"] > 0))
        max_depth.append(
            float(torch.stack(shortfalls, dim=-1).max()) if shortfalls else 0.0
        )

    task = np.asarray(task_loss, dtype=float)
    control = np.asarray(effort, dtype=float)
    memory = policy_memory_footprint(policy)
    return {
        "n_scenarios": int(len(task)),
        "task_loss_mean": float(task.mean()),
        "task_loss_std": float(task.std()),
        "control_effort_mean": float(control.mean()),
        "control_effort_std": float(control.std()),
        "scenario_violation_percent": float(100.0 * np.mean(violation)),
        "maximum_violation_depth": float(np.max(max_depth)),
        "policy_sparse_indexed_kib": float(memory["policy_sparse_indexed_kib_int32"]),
    }


_REFERENCE_RESULTS = {
    "TwoTank": {"task": (0.209, 0.039), "violations": 0.0, "memory": 0.047},
    "VanDerPol": {"task": (96.8, 183.8), "violations": 0.0, "memory": 0.023},
    "DoubleIntegrator": {
        "task": (62.6, 58.0), "effort": (12.07, 1.03),
        "violations": 60.8, "memory": 0.094,
    },
}


def validate_reference_results(case: str, metrics: Dict) -> Dict:
    """Check task, effort, safety, and memory at the table's displayed precision."""
    expected = _REFERENCE_RESULTS[case]
    actual = {
        "task": (1000.0 * metrics["task_loss_mean"], 1000.0 * metrics["task_loss_std"]),
        "violations": metrics["scenario_violation_percent"],
        "memory": metrics["policy_sparse_indexed_kib"],
    }
    if "effort" in expected:
        actual["effort"] = (metrics["control_effort_mean"], metrics["control_effort_std"])
    task_tolerance = 0.001 if case == "TwoTank" else 0.1
    passed = all(
        abs(value - target) <= task_tolerance
        for value, target in zip(actual["task"], expected["task"])
    )
    passed = passed and abs(actual["violations"] - expected["violations"]) <= 0.05
    passed = passed and abs(actual["memory"] - expected["memory"]) <= 0.0005
    if "effort" in expected:
        passed = passed and all(
            abs(value - target) <= 0.005
            for value, target in zip(actual["effort"], expected["effort"])
        )
    return {"passed": passed, "actual": actual, "expected": expected}


def save_distilled_policy(path: Path | str, policy: DistilledPolicy) -> None:
    """Save supports and per-output coefficients without serializing Python objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "active_idx": policy.active_idx,
            "coefficients": [coefficient.detach().cpu() for coefficient in policy.Xi],
        },
        path,
    )


def load_distilled_policy(source_policy, path: Path | str, device="cpu") -> DistilledPolicy:
    """Load a distilled controller using the source policy's dictionary."""
    saved = torch.load(path, map_location=device, weights_only=True)
    coefficients = [value.reshape(-1).tolist() for value in saved["coefficients"]]
    return DistilledPolicy(
        source_policy.library, saved["active_idx"], coefficients, device=device,
    ).eval()
