"""Evaluation metrics for controller-comparison tables.

All functions take closed-loop trajectories and return plain floats/ints so results can be
aggregated across seeds and dumped to CSV/JSON. Definitions:

* tracking MSE            mean ||x_t - r_t||^2
* control cost (L1/L2)    sum_t ||u_t||_1  and  sum_t ||u_t||_2^2
* control smoothness      RMS / mean / max ||u_t-u_{t-1}|| and total variation
* violation magnitude     sum over steps/constraints of max(0, delta - h_i(x_t))
* num violations          number of (step, constraint) pairs with h_i < delta
* min safety margin       min_t,i h_i(x_t)  (larger is safer)
* sparsity                number of active policy terms (per output and total)
* policy memory           learned parameter/buffer tensor storage in bytes
"""
from __future__ import annotations

import math
import statistics
import time
from typing import Dict, List, Optional

import torch

from ..safety.specs import SafetySpec

__all__ = [
    "tracking_mse",
    "terminal_tracking",
    "control_cost",
    "control_smoothness",
    "violation_stats",
    "policy_sparsity",
    "policy_memory_footprint",
    "policy_forward_latency",
    "compute_metrics",
]


def _as_bt(x: torch.Tensor) -> torch.Tensor:
    return x if x.dim() == 3 else x.unsqueeze(0)


def tracking_mse(x_traj: torch.Tensor, r_traj: torch.Tensor) -> float:
    x, r = _as_bt(x_traj), _as_bt(r_traj)
    T = min(x.shape[1], r.shape[1])
    return float(((x[:, :T] - r[:, :T]) ** 2).sum(dim=-1).mean().item())


def terminal_tracking(
    x_traj: torch.Tensor,
    r_traj: torch.Tensor,
    *,
    reach_tolerance: float = 0.05,
    reach_hold_steps: int = 5,
) -> Dict[str, float]:
    """Terminal accuracy and sustained arrival time for the final reference segment.

    ``steps_to_reach`` starts at the real step where the final reference becomes active.
    Arrival is a settling time: the Euclidean state error must be within
    ``reach_tolerance`` for the final ``reach_hold_steps`` samples, and the reported step
    is the start of the final uninterrupted in-tolerance interval. ``steps_to_reach`` is
    relative to the final reference change; ``final_reference_stabilization_step`` is its
    absolute trajectory index. Both are NaN when sustained stabilization is not achieved.
    """
    x, r = _as_bt(x_traj), _as_bt(r_traj)
    T = min(x.shape[1], r.shape[1])
    if T < 1:
        raise ValueError("terminal tracking requires a nonempty trajectory")
    if reach_tolerance < 0.0 or reach_hold_steps < 1:
        raise ValueError("reach_tolerance must be nonnegative and reach_hold_steps positive")

    final_error = x[:, T - 1] - r[:, T - 1]
    final_mse = (final_error ** 2).mean(dim=-1)
    final_l2 = torch.linalg.vector_norm(final_error, dim=-1)
    reached = []
    segment_starts = []
    arrival_steps = []
    stabilization_steps = []
    for batch in range(x.shape[0]):
        final_reference = r[batch, T - 1]
        same_reference = torch.isclose(
            r[batch, :T], final_reference.unsqueeze(0), rtol=0.0, atol=1.0e-7
        ).all(dim=-1)
        segment_start = T - 1
        while segment_start > 0 and bool(same_reference[segment_start - 1].item()):
            segment_start -= 1
        segment_starts.append(float(segment_start))
        error = torch.linalg.vector_norm(
            x[batch, segment_start:T] - final_reference.unsqueeze(0), dim=-1
        )
        inside = error <= float(reach_tolerance)
        hold = min(int(reach_hold_steps), inside.numel())
        arrival = None
        if bool(inside[-hold:].all().item()):
            arrival = inside.numel() - 1
            while arrival > 0 and bool(inside[arrival - 1].item()):
                arrival -= 1
        reached.append(arrival is not None)
        if arrival is not None:
            arrival_steps.append(float(arrival))
            stabilization_steps.append(float(segment_start + arrival))

    return {
        "final_tracking_mse": float(final_mse.mean().item()),
        "final_tracking_error_l2": float(final_l2.mean().item()),
        "reached_reference": float(sum(reached) / len(reached)),
        "steps_to_reach": (
            float(sum(arrival_steps) / len(arrival_steps)) if arrival_steps else float("nan")
        ),
        "final_reference_start_step": float(sum(segment_starts) / len(segment_starts)),
        "final_reference_stabilization_step": (
            float(sum(stabilization_steps) / len(stabilization_steps))
            if stabilization_steps else float("nan")
        ),
        "reach_tolerance": float(reach_tolerance),
        "reach_hold_steps": int(reach_hold_steps),
    }


def control_cost(u_traj: torch.Tensor) -> Dict[str, float]:
    u = _as_bt(u_traj)
    return {
        "control_cost_l1": float(u.abs().sum(dim=-1).sum(dim=1).mean().item()),
        "control_cost_l2": float((u ** 2).sum(dim=-1).sum(dim=1).mean().item()),
        "total_control_action_l1": float(u.abs().sum(dim=-1).sum(dim=1).mean().item()),
    }


def control_smoothness(u_traj: torch.Tensor) -> Dict[str, float]:
    """Control-increment metrics; smaller values mean smoother applied actions."""
    u = _as_bt(u_traj)
    if u.shape[1] < 2:
        return {
            "control_smoothness_rms": 0.0,
            "mean_du_l2": 0.0,
            "max_du_l2": 0.0,
            "control_total_variation_l1": 0.0,
        }
    du = u[:, 1:, :] - u[:, :-1, :]
    du_l2 = torch.linalg.vector_norm(du, dim=-1)
    return {
        "control_smoothness_rms": float(torch.sqrt((du ** 2).sum(dim=-1).mean()).item()),
        "mean_du_l2": float(du_l2.mean().item()),
        "max_du_l2": float(du_l2.max().item()),
        "control_total_variation_l1": float(du.abs().sum(dim=-1).sum(dim=1).mean().item()),
    }


def violation_stats(
    x_traj: torch.Tensor,
    spec: SafetySpec,
    u_traj: Optional[torch.Tensor] = None,
    *,
    tol: float = 1.0e-7,
) -> Dict[str, float]:
    """State/control-rate violations with a small numerical feasibility tolerance."""
    x = _as_bt(x_traj)
    total_mag = 0.0
    num_viol = 0
    min_margin = float("inf")
    for con in spec.state_constraints:
        h = con.margin(x)
        min_margin = min(min_margin, float(h.min().item()))
        short = torch.clamp(con.delta - tol - h, min=0.0)
        total_mag += float(short.sum().item())
        num_viol += int((h < con.delta - tol).sum().item())
    out = {
        "violation_magnitude": total_mag,
        "num_state_violations": num_viol,
        "num_violations": num_viol,
        "min_state_margin": (min_margin if min_margin != float("inf") else float("nan")),
        "min_safety_margin": (min_margin if min_margin != float("inf") else float("nan")),
    }
    if spec.control_rate is not None and u_traj is not None:
        u = _as_bt(u_traj)
        if u.shape[1] > 1:
            rate = spec.control_rate
            du_sq = ((u[:, 1:, :] - u[:, :-1, :]) ** 2).sum(dim=-1)
            h_du = rate.margin(du_sq)
            short = torch.clamp(rate.delta - tol - h_du, min=0.0)
            rate_viol = int((h_du < rate.delta - tol).sum().item())
            rate_min = float((h_du - rate.delta).min().item())
            out["control_rate_violation_magnitude"] = float(short.sum().item())
            out["num_control_rate_violations"] = rate_viol
            out["min_control_rate_margin"] = rate_min
            out["violation_magnitude"] += out["control_rate_violation_magnitude"]
            out["num_violations"] += rate_viol
            out["min_safety_margin"] = min(out["min_safety_margin"], rate_min)
        else:
            out["control_rate_violation_magnitude"] = 0.0
            out["num_control_rate_violations"] = 0
            out["min_control_rate_margin"] = float("nan")
    return out


def policy_sparsity(policy) -> Dict[str, float]:
    per_output = [int(p.shape[0]) for p in policy.Xi]
    return {
        "sparsity_total_terms": int(sum(per_output)),
        "sparsity_per_output": per_output,
    }


def policy_memory_footprint(policy) -> Dict[str, float]:
    """Controller tensor storage, independent of Python and serialization overhead.

    Parameter bytes are the fairest architecture-level comparison. Buffer bytes cover
    registered non-parameter tensors. For sparse policies, the indexed estimate adds one
    32-bit term identifier per active coefficient; executable/framework memory is excluded.
    """
    parameters = list(policy.parameters()) if hasattr(policy, "parameters") else []
    buffers = list(policy.buffers()) if hasattr(policy, "buffers") else []
    parameter_count = int(sum(parameter.numel() for parameter in parameters))
    parameter_bytes = int(sum(parameter.numel() * parameter.element_size() for parameter in parameters))
    buffer_bytes = int(sum(buffer.numel() * buffer.element_size() for buffer in buffers))
    out = {
        "policy_parameter_count": parameter_count,
        "policy_parameter_bytes": parameter_bytes,
        "policy_parameter_kib": parameter_bytes / 1024.0,
        "policy_buffer_bytes": buffer_bytes,
        "policy_tensor_state_bytes": parameter_bytes + buffer_bytes,
        "policy_tensor_state_kib": (parameter_bytes + buffer_bytes) / 1024.0,
    }
    if hasattr(policy, "Xi"):
        index_bytes = 4 * int(sum(parameter.numel() for parameter in policy.Xi))
        out["policy_sparse_term_index_bytes_int32"] = index_bytes
        out["policy_sparse_indexed_bytes_int32"] = parameter_bytes + index_bytes
        out["policy_sparse_indexed_kib_int32"] = (parameter_bytes + index_bytes) / 1024.0
    return out


def policy_forward_latency(
    policy,
    x_traj: torch.Tensor,
    r_traj: torch.Tensor,
    *,
    warmup: int = 10,
    repeats: int = 200,
    iters: int = 1000,
    batch_size: int = 1,
    seed: int = 0,
    compiled: bool = False,
    compile_backend: Optional[str] = None,
    compile_mode: Optional[str] = None,
    compile_fullgraph: bool = False,
    compile_dynamic: bool = False,
) -> Dict[str, float]:
    """Mean batch-1 latency of only ``policy(X, R)``.

    Random inputs, warm-up calls, synchronization, rollout, clamping, dynamics,
    and safety logic are outside the timed region. Each reported sample
    is the elapsed time of ``iters`` policy calls divided by ``iters``; the final value
    is the mean across ``repeats`` independently generated input pairs. When
    ``compiled=True``, ``torch.compile`` is called once and its lazy compilation occurs
    during the untimed warm-up calls.
    """
    x, r = _as_bt(x_traj), _as_bt(r_traj)
    if warmup < 0 or repeats < 1 or iters < 1 or batch_size < 1:
        raise ValueError("warmup must be nonnegative; repeats, iters, and batch_size must be positive")

    def synchronize():
        if x.is_cuda:
            torch.cuda.synchronize(x.device)

    generator = torch.Generator(device=x.device).manual_seed(int(seed))
    was_training = getattr(policy, "training", None)
    if hasattr(policy, "eval"):
        policy.eval()
    forward_policy = policy
    if compiled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("compiled policy timing requires torch.compile")
        compile_kwargs = {
            "fullgraph": bool(compile_fullgraph),
            "dynamic": bool(compile_dynamic),
        }
        if compile_backend is not None:
            compile_kwargs["backend"] = str(compile_backend)
        if compile_mode is not None:
            compile_kwargs["mode"] = str(compile_mode)
        forward_policy = torch.compile(policy, **compile_kwargs)
    per_call = []
    with torch.no_grad():
        for _ in range(int(repeats)):
            X = torch.rand(
                int(batch_size), x.shape[-1], device=x.device,
                dtype=x.dtype, generator=generator,
            )
            R = torch.rand(
                int(batch_size), r.shape[-1], device=r.device,
                dtype=r.dtype, generator=generator,
            )
            for _ in range(int(warmup)):
                forward_policy(X, R)
            synchronize()
            t0 = time.perf_counter()
            for _ in range(int(iters)):
                forward_policy(X, R)
            synchronize()
            per_call.append((time.perf_counter() - t0) / int(iters))
    if was_training and hasattr(policy, "train"):
        policy.train()

    latency_s = statistics.mean(per_call)
    return {
        "policy_forward_per_step_s": latency_s,
        "policy_forward_per_step_ms": 1.0e3 * latency_s,
        "policy_forward_per_step_us": 1.0e6 * latency_s,
        "policy_forward_compiled": float(bool(compiled)),
    }


def compute_metrics(
    x_traj: torch.Tensor,
    u_traj: torch.Tensor,
    r_traj: torch.Tensor,
    *,
    spec: Optional[SafetySpec] = None,
    policy=None,
    reach_tolerance: float = 0.05,
    reach_hold_steps: int = 5,
) -> Dict[str, float]:
    """Aggregate every available metric for one closed-loop run into a flat dict."""
    m: Dict[str, float] = {"tracking_mse": tracking_mse(x_traj, r_traj)}
    m.update(terminal_tracking(
        x_traj, r_traj,
        reach_tolerance=reach_tolerance,
        reach_hold_steps=reach_hold_steps,
    ))
    m.update(control_cost(u_traj))
    m.update(control_smoothness(u_traj))
    if spec is not None:
        m.update(violation_stats(x_traj, spec, u_traj))
        m["success"] = int(m.get("num_violations", 0) == 0)
    if policy is not None:
        m.update(policy_memory_footprint(policy))
    if policy is not None and hasattr(policy, "Xi"):
        m.update(policy_sparsity(policy))
    elif policy is not None and hasattr(policy, "parameters"):
        if hasattr(policy, "dpc_loss_reference_verified"):
            verified = policy.dpc_loss_reference_verified
            m["dpc_loss_reference_verified"] = -1 if verified is None else int(bool(verified))
    return m
