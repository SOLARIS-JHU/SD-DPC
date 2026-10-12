"""TwoTank controller and safety benchmarks used by the Stage 5 notebook."""
from __future__ import annotations

import statistics
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from ..baselines import solve_mpc
from .evaluate import aggregate
from .metrics import compute_metrics, policy_forward_latency
from .rollout import rollout_closed_loop

__all__ = [
    "make_twotank_scenarios",
    "evaluate_twotank_nominal",
    "violation_summary",
]


def _equal_reference_batch(levels: torch.Tensor, nsteps: int, nx: int) -> torch.Tensor:
    """Expand levels into equal control-step segments plus one terminal sample."""
    n_traj, n_refs = levels.shape
    if n_refs < 1 or n_refs > nsteps:
        raise ValueError("n_references must be between 1 and nsteps")
    base, remainder = divmod(int(nsteps), int(n_refs))
    holds = [base + (i < remainder) for i in range(n_refs)]
    chunks = [
        levels[:, i].reshape(n_traj, 1, 1).expand(n_traj, hold, nx)
        for i, hold in enumerate(holds)
    ]
    control_references = torch.cat(chunks, dim=1)
    return torch.cat([control_references, control_references[:, -1:, :]], dim=1).contiguous()


def _configured_state_bounds(system, cfg: Dict) -> Tuple[float, float]:
    """Resolve evaluation bounds while retaining the legacy safe_xmin/xmax keys."""
    xmin = float(cfg.get("xmin", cfg.get("safe_xmin", system.xmin)))
    xmax = float(cfg.get("xmax", cfg.get("safe_xmax", system.xmax)))
    if not float(system.xmin) <= xmin < xmax <= float(system.xmax):
        raise ValueError(
            f"TwoTank bounds must satisfy {system.xmin} <= xmin < xmax <= {system.xmax}; "
            f"got xmin={xmin}, xmax={xmax}"
        )
    return xmin, xmax


def _fraction_levels(values, xmin: float, xmax: float, *, name: str, device) -> torch.Tensor:
    fractions = torch.as_tensor(values, dtype=torch.float32, device=device).flatten()
    if fractions.numel() == 0 or bool(((fractions < 0.0) | (fractions > 1.0)).any()):
        raise ValueError(f"{name} must contain fractions in [0, 1]")
    return xmin + (xmax - xmin) * fractions


def _required_box_clearance(cfg: Dict) -> float:
    box_band = float(cfg.get("bands", {}).get("box", 0.0))
    if box_band < 0.0:
        raise ValueError("TwoTank box band must be nonnegative")
    return box_band


def _validate_reference_clearance(levels: torch.Tensor, xmin: float, xmax: float, cfg: Dict):
    clearance = _required_box_clearance(cfg)
    span = xmax - xmin
    overlap_tol = 1.0e-12 * max(1.0, span)
    if 2.0 * clearance >= span - overlap_tol:
        raise ValueError(
            f"box clearance {clearance} overlaps across bounds [{xmin}, {xmax}]; "
            f"it must be smaller than {0.5 * span}"
        )
    allowed_min, allowed_max = xmin + clearance, xmax - clearance
    tol = 1.0e-7 * max(1.0, span)
    actual_min = float(levels.min().item())
    actual_max = float(levels.max().item())
    if actual_min < allowed_min - tol or actual_max > allowed_max + tol:
        raise ValueError(
            f"references [{actual_min:.6g}, {actual_max:.6g}] conflict with box "
            f"clearance {clearance}; choose references inside "
            f"[{allowed_min:.6g}, {allowed_max:.6g}]"
        )


def make_twotank_scenarios(system, cfg: Dict, device=None) -> Dict[str, torch.Tensor]:
    """Create different starts and equal-duration scalar reference schedules.

    ``reference.mode='random'`` samples every level independently. ``'safety_edges'``
    inserts configured near-boundary levels and fills the remaining slots from the
    interior range, then independently shuffles each trajectory's schedule.
    ``'sequence'`` applies the configured scalar ``reference.levels`` in order to
    every trajectory; each scalar is used for every state. Fraction ranges are
    mapped relative to the configured ``xmin``/``xmax`` limits.
    """
    device = device or torch.device("cpu")
    n_traj = int(cfg.get("n_trajectories", 50))
    nsteps = int(cfg.get("nsteps", 1000))
    n_refs = int(cfg.get("n_references", 4))
    if n_traj < 1 or nsteps < 1 or n_refs < 1:
        raise ValueError("n_trajectories, nsteps, and n_references must be positive")

    xmin, xmax = _configured_state_bounds(system, cfg)
    span = xmax - xmin
    generator = torch.Generator(device=device).manual_seed(int(cfg.get("seed", 0)))
    if "initial_fraction_range" in cfg:
        initial_range = cfg["initial_fraction_range"]
        if len(initial_range) != 2:
            raise ValueError("initial_fraction_range must be [min_fraction, max_fraction]")
        xlo, xhi = xmin + span * float(initial_range[0]), xmin + span * float(initial_range[1])
    else:
        xlo = float(cfg.get("x0_min", xmin + 0.1 * span))
        xhi = float(cfg.get("x0_max", xmin + 0.9 * span))
    if not xmin <= xlo < xhi <= xmax:
        raise ValueError(f"initial-state range [{xlo}, {xhi}] must lie inside [{xmin}, {xmax}]")
    configured_x0 = cfg.get("initial_state")
    if configured_x0 is None:
        x0 = xlo + (xhi - xlo) * torch.rand(
            n_traj, system.nx, generator=generator, device=device
        )
    else:
        initial = torch.as_tensor(
            configured_x0, dtype=torch.float32, device=device
        ).flatten()
        if initial.numel() != system.nx:
            raise ValueError(
                f"initial_state must contain {system.nx} values; got {initial.numel()}"
            )
        if bool(((initial < xmin) | (initial > xmax)).any()):
            raise ValueError(f"initial_state must lie inside [{xmin}, {xmax}]")
        x0 = initial.reshape(1, system.nx).expand(n_traj, -1).clone()

    ref_cfg = cfg.get("reference", {})
    mode = ref_cfg.get("mode", "random")
    if mode == "random":
        if "fraction_range" in ref_cfg:
            fraction_range = ref_cfg["fraction_range"]
            if len(fraction_range) != 2:
                raise ValueError("reference.fraction_range must have two values")
            lo = xmin + span * float(fraction_range[0])
            hi = xmin + span * float(fraction_range[1])
        else:
            lo = float(ref_cfg.get("min", xmin + 0.15 * span))
            hi = float(ref_cfg.get("max", xmin + 0.85 * span))
        if not xmin <= lo < hi <= xmax:
            raise ValueError(f"reference range [{lo}, {hi}] must lie inside [{xmin}, {xmax}]")
        _validate_reference_clearance(
            torch.tensor([lo, hi], dtype=torch.float32, device=device), xmin, xmax, cfg
        )
        levels = lo + (hi - lo) * torch.rand(
            n_traj, n_refs, generator=generator, device=device
        )
    elif mode == "safety_edges":
        if "edge_fractions" in ref_cfg:
            edges = _fraction_levels(
                ref_cfg["edge_fractions"], xmin, xmax,
                name="reference.edge_fractions", device=device,
            )
        else:
            edges = torch.as_tensor(
                ref_cfg.get("edge_levels", [xmin + 0.05 * span, xmin + 0.95 * span]),
                dtype=torch.float32, device=device,
            ).flatten()
        if edges.numel() > n_refs:
            raise ValueError("configured edge references cannot exceed n_references")
        n_interior = n_refs - edges.numel()
        if "interior_fraction_range" in ref_cfg:
            interior_range = ref_cfg["interior_fraction_range"]
            if len(interior_range) != 2:
                raise ValueError("reference.interior_fraction_range must have two values")
            lo = xmin + span * float(interior_range[0])
            hi = xmin + span * float(interior_range[1])
        else:
            lo = float(ref_cfg.get("interior_min", xmin + 0.2 * span))
            hi = float(ref_cfg.get("interior_max", xmin + 0.8 * span))
        if not xmin <= lo <= hi <= xmax:
            raise ValueError(f"interior reference range [{lo}, {hi}] must lie inside [{xmin}, {xmax}]")
        if n_interior:
            _validate_reference_clearance(
                torch.tensor([lo, hi], dtype=torch.float32, device=device), xmin, xmax, cfg
            )
        interior = lo + (hi - lo) * torch.rand(
            n_traj, n_interior, generator=generator, device=device
        )
        levels = torch.cat([edges.reshape(1, -1).expand(n_traj, -1), interior], dim=1)
        if bool(ref_cfg.get("shuffle", True)):
            order = torch.rand(n_traj, n_refs, generator=generator, device=device).argsort(dim=1)
            levels = torch.gather(levels, 1, order)
    elif mode == "sequence":
        # A scalar tank-level reference is shared by x1 and x2. The expansion to
        # both states and to equal-duration horizon segments happens below.
        configured_levels = torch.as_tensor(
            ref_cfg.get("levels", []), dtype=torch.float32, device=device
        ).flatten()
        if configured_levels.numel() < 1:
            raise ValueError("reference.levels must contain at least one scalar")
        if configured_levels.numel() > nsteps:
            raise ValueError("reference.levels cannot contain more values than nsteps")
        levels = configured_levels.reshape(1, -1).expand(n_traj, -1).clone()
    else:
        raise ValueError(f"unknown TwoTank reference mode {mode!r}")

    _validate_reference_clearance(levels, xmin, xmax, cfg)
    r = _equal_reference_batch(levels, nsteps, system.nx)
    bounds = torch.tensor([xmin, xmax], dtype=torch.float32, device=device)
    return {
        "xn": x0.unsqueeze(1), "r": r,
        "reference_levels": levels, "state_bounds": bounds,
    }
def _single_scenarios(data: Dict[str, torch.Tensor]) -> Iterable[Dict[str, torch.Tensor]]:
    for i in range(data["xn"].shape[0]):
        scenario = {"xn": data["xn"][i : i + 1], "r": data["r"][i : i + 1]}
        if "reference_levels" in data:
            scenario["reference_levels"] = data["reference_levels"][i : i + 1]
        yield scenario


def _trajectory_rows(
    result: Dict,
    data: Dict[str, torch.Tensor],
    spec,
    *,
    method: str,
    extra: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    rows = []
    for i in range(result["x_traj"].shape[0]):
        metrics = compute_metrics(
            result["x_traj"][i : i + 1],
            result["u_traj"][i : i + 1],
            data["r"][i : i + 1],
            spec=spec,
        )
        metrics.update(extra or {})
        for j, value in enumerate(data["xn"][i, 0].detach().cpu().tolist()):
            metrics[f"initial_x{j}"] = float(value)
        if "reference_levels" in data:
            for j, value in enumerate(data["reference_levels"][i].detach().cpu().tolist()):
                metrics[f"reference_{j}"] = float(value)
        metrics["method"] = method
        metrics["trajectory"] = i
        rows.append(metrics)
    return rows


def violation_summary(per_trajectory: Dict[str, List[Dict]]) -> Dict[str, Dict[str, float]]:
    """Summarize violation counts across trajectories, not only across time steps."""
    out = {}
    for method, rows in per_trajectory.items():
        counts = [int(row.get("num_violations", 0)) for row in rows]
        state = [int(row.get("num_state_violations", 0)) for row in rows]
        rate = [int(row.get("num_control_rate_violations", 0)) for row in rows]
        n_bad = sum(count > 0 for count in counts)
        out[method] = {
            "n_trajectories": len(counts),
            "trajectories_with_violations": n_bad,
            "trajectory_violation_fraction": n_bad / max(len(counts), 1),
            "total_violations": sum(counts),
            "mean_violations_per_trajectory": statistics.mean(counts) if counts else 0.0,
            "median_violations_per_trajectory": statistics.median(counts) if counts else 0.0,
            "max_violations_per_trajectory": max(counts, default=0),
            "total_state_violations": sum(state),
            "total_control_rate_violations": sum(rate),
        }
    return out


def evaluate_twotank_nominal(
    system,
    sparse_policy,
    nn_policy,
    cfg: Dict,
    device=None,
    *,
    progress: bool = True,
    policy_latencies: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict:
    """Evaluate the controllers selected by ``cfg['methods']`` on shared scenarios."""
    device = device or torch.device("cpu")
    methods = list(cfg.get("methods", ["sd_dpc", "nn_dpc", "mpc"]))
    allowed = {"sd_dpc", "nn_dpc", "mpc"}
    unknown = sorted(set(methods) - allowed)
    if not methods or unknown:
        raise ValueError(
            "methods must be a nonempty selection from ['sd_dpc', 'nn_dpc', 'mpc']; "
            f"got {methods}"
        )
    if "nn_dpc" in methods and nn_policy is None:
        raise ValueError("method 'nn_dpc' requires an NN-DPC policy")

    data = make_twotank_scenarios(system, cfg, device)
    spec = system.safety_specs(cfg)
    plant = system.perturbed_plant({})
    scale = float(cfg.get("action_scale", 1.0))
    per_trajectory = {name: [] for name in methods}
    trajectories = {
        name: {"x_traj": [], "u_traj": []}
        for name in per_trajectory
    }
    examples = {}
    controllers = {"sd_dpc": sparse_policy, "nn_dpc": nn_policy}
    for name in (method for method in methods if method in controllers):
        policy = controllers[name]
        result = rollout_closed_loop(
            policy, plant, data, umin=system.umin, umax=system.umax, action_scale=scale
        )
        latency = (policy_latencies or {}).get(name)
        if latency is None:
            latency = policy_forward_latency(
                policy, result["x_traj"][:1], data["r"][:1],
                **cfg.get("inference_benchmark", {}),
            )
        extra = {
            **latency,
            "batched_rollout_wall_time_s": float(result["rollout_time_s"]),
        }
        per_trajectory[name] = _trajectory_rows(
            result, data, spec, method=name, extra=extra
        )
        trajectories[name] = {
            "x_traj": result["x_traj"].detach().cpu(),
            "u_traj": result["u_traj"].detach().cpu(),
        }
        examples[name] = {
            "x_traj": result["x_traj"][:1], "u_traj": result["u_traj"][:1]
        }

    # Allow MPC-only objective weights without changing the fixed DPC policy's
    # recorded loss configuration. Nested MPC weights override only named keys.
    mpc_overrides = cfg.get("mpc", {})
    mpc_cfg = {**cfg, **mpc_overrides}
    mpc_cfg["weights"] = {
        **cfg.get("weights", {}),
        **mpc_overrides.get("weights", {}),
    }
    for i, scenario in enumerate(_single_scenarios(data) if "mpc" in methods else ()):
        if progress:
            print(f"MPC trajectory {i + 1}/{data['xn'].shape[0]}")
        result = solve_mpc(system, scenario, mpc_cfg)
        if progress:
            print(
                f"  {result['solver_status']}: horizon={result.get('mpc_horizon', cfg['nsteps'])}, "
                f"mean {result.get('solver_iterations_mean', result['solver_iterations']):.1f} "
                f"iterations/step, {result['solve_time_s']:.3f} s total"
            )
        extra = {
            "mpc_solve_time_s": float(result["solve_time_s"]),
            "mpc_solve_per_step_ms": 1.0e3 * float(result["per_step_s"]),
            "mpc_solver_iterations": int(result["solver_iterations"]),
            "mpc_iterations_per_solve": float(
                result.get("solver_iterations_mean", result["solver_iterations"])
            ),
            "mpc_max_iterations_per_solve": int(
                result.get("solver_iterations_max", result["solver_iterations"])
            ),
            "mpc_setup_time_s": float(result.get("setup_time_s", 0.0)),
            "mpc_horizon": int(result.get("mpc_horizon", cfg["nsteps"])),
        }
        row = _trajectory_rows(result, scenario, spec, method="mpc", extra=extra)[0]
        row["trajectory"] = i
        per_trajectory["mpc"].append(row)
        trajectories["mpc"]["x_traj"].append(result["x_traj"].detach().cpu())
        trajectories["mpc"]["u_traj"].append(result["u_traj"].detach().cpu())
        if i == 0:
            examples["mpc"] = {
                "x_traj": result["x_traj"], "u_traj": result["u_traj"]
            }

    if "mpc" in methods:
        trajectories["mpc"] = {
            key: torch.cat(values, dim=0)
            for key, values in trajectories["mpc"].items()
        }

    plain = {
        method: [{k: v for k, v in row.items() if k not in {"method", "trajectory"}}
                 for row in rows]
        for method, rows in per_trajectory.items()
    }
    return {
        "scenarios": data,
        "per_trajectory": per_trajectory,
        "summary": aggregate(plain),
        "violations": violation_summary(per_trajectory),
        "trajectories": trajectories,
        "examples": examples,
    }
