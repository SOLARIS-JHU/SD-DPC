"""Closed-loop rollout runners for fixed-policy evaluation."""
from __future__ import annotations

import time
from typing import Dict, Optional

import torch

from ..actions import bounded_action

__all__ = ["rollout_closed_loop", "make_test_data"]


def rollout_closed_loop(
    policy,
    plant,
    data: Dict[str, torch.Tensor],
    *,
    umin=None,
    umax=None,
    action_scale: float = 1.0,
) -> Dict:
    """Roll ``policy`` through ``plant`` following ``data['r']`` from ``data['xn']``.

    Returns ``x_traj`` (B, T, nx), ``u_traj`` (B, T-1, nu) and per-step wall-clock timing.
    """
    x0 = data["xn"][:, 0, :]
    r = data["r"]
    T = r.shape[1]
    t0 = time.perf_counter()
    xs, us = [x0.unsqueeze(1)], []
    with torch.no_grad():
        x = x0
        for step in range(T - 1):
            u = bounded_action(policy(x, r[:, step, :]), umin, umax, action_scale)
            us.append(u.unsqueeze(1))
            x = plant(x, u)
            xs.append(x.unsqueeze(1))
    x_traj, u_traj = torch.cat(xs, dim=1), torch.cat(us, dim=1)
    elapsed = time.perf_counter() - t0
    return {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "r_traj": r,
        "rollout_time_s": elapsed,
        "per_step_s": elapsed / max(T - 1, 1),
    }


def make_test_data(
    nx: int,
    nsteps: int,
    x0,
    ref,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build a single-trajectory ``{'xn','r'}`` dict from an initial state and a reference.

    ``ref`` may be a constant (broadcast over the horizon) or a full (nsteps+1, nref) array.
    """
    device = device or torch.device("cpu")
    xn = torch.as_tensor(x0, dtype=torch.float32, device=device).reshape(1, 1, nx)
    ref_t = torch.as_tensor(ref, dtype=torch.float32, device=device)
    if ref_t.dim() == 1:
        ref_t = ref_t.reshape(1, 1, -1).expand(1, nsteps + 1, ref_t.numel()).contiguous()
    else:
        ref_t = ref_t.reshape(1, ref_t.shape[0], ref_t.shape[1])
    return {"xn": xn, "r": ref_t}
