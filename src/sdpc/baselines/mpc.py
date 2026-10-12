"""Optimal-control (MPC) baseline via CasADi/IPOPT (Sec. 5 comparisons).

Supports DPC-matched open-loop blocks and an optional online receding-horizon solve using
the system's CasADi dynamics hooks. A DPC-matched block optimizes the complete policy-
training horizon and applies that plan without re-optimizing at every control step.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

try:  # pragma: no cover - environment dependent
    import casadi as ca

    _HAS_CASADI = True
except Exception:  # pragma: no cover
    ca = None
    _HAS_CASADI = False

__all__ = ["mpc_available", "solve_mpc"]


def mpc_available() -> bool:
    return _HAS_CASADI


def _numpy_guess(value, expected_shape, name):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value, dtype=float)
    if value.ndim == len(expected_shape) + 1 and value.shape[0] == 1:
        value = value[0]
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")
    return value


def _ipopt_options(cfg: Dict) -> Dict:
    tol = float(cfg.get("tol", 1.0e-6))
    options = {
        "print_level": int(cfg.get("print_level", 0)),
        "sb": "yes",
        "max_iter": int(cfg.get("max_iter", 2000)),
        "tol": tol,
        "acceptable_tol": float(cfg.get("acceptable_tol", max(100.0 * tol, 1.0e-4))),
        "acceptable_iter": int(cfg.get("acceptable_iter", 15)),
        "mu_strategy": cfg.get("mu_strategy", "adaptive"),
        "hessian_approximation": cfg.get("hessian_approximation", "exact"),
        "bound_push": float(cfg.get("bound_push", 1.0e-6)),
        "bound_frac": float(cfg.get("bound_frac", 1.0e-6)),
    }
    if bool(cfg.get("warm_start", False)):
        options["warm_start_init_point"] = "yes"
    return options


def _held_reference(reference: np.ndarray, step: int, horizon: int) -> np.ndarray:
    """Return only the reference active at ``step``, repeated over an MPC horizon."""
    current = np.asarray(reference[step], dtype=float).reshape(1, -1)
    return np.repeat(current, int(horizon) + 1, axis=0)


def _next_reference_change_horizon(reference: np.ndarray, step: int, maximum: int) -> int:
    """Return ``min(maximum, steps until the next reference change)``.

    Thus with a change at step 250 and ``maximum=50``, the horizon is 50 through
    step 200, then 49 at step 201 down to 1 at step 249.  The solver only sees
    the currently active reference; the known switch time schedules the horizon.
    """
    current = np.asarray(reference[step], dtype=float)
    later = np.asarray(reference[step + 1:], dtype=float)
    if later.size == 0:
        return int(maximum)
    changed = np.flatnonzero(~np.isclose(later, current, rtol=0.0, atol=1.0e-9).all(axis=1))
    remaining = int(changed[0] + 1) if changed.size else int(maximum)
    return max(1, min(int(maximum), remaining))


def _solve_reference_change_horizon_mpc(system, data, cfg: Dict, hooks: Dict) -> Dict:
    """Receding-horizon MPC whose horizon ends at the next known reference switch.

    A separate cached NLP is built for each required horizon (1 through the
    configured maximum), then warm-started when that horizon occurs again.  This
    avoids changing the fixed-horizon baseline unless the schedule is explicit.
    """
    import time as _time

    if bool(cfg.get("arrival_deadline", False)):
        raise ValueError("arrival_deadline is not supported with horizon_schedule='next_reference_change'")
    f_casadi = hooks["f_casadi"]
    h_casadi = hooks.get("h_casadi", lambda x: [])
    x_initial = data["xn"][0, 0, :].detach().cpu().numpy().astype(float)
    reference = data["r"][0].detach().cpu().numpy().astype(float)
    T = reference.shape[0]
    nx, nu = system.nx, system.nu
    maximum = int(cfg.get("horizon", cfg.get("nsteps", 50)))
    if maximum < 1:
        raise ValueError("MPC horizon must be positive")
    make_u_guess = hooks.get("mpc_u_guess")
    cache = {}
    setup_time = 0.0

    def build_problem(horizon: int):
        nonlocal setup_time
        started = _time.perf_counter()
        opti = ca.Opti()
        X = opti.variable(nx, horizon + 1)
        U = opti.variable(nu, horizon)
        x0_param = opti.parameter(nx)
        ref_param = opti.parameter(nx)
        opti.subject_to(X[:, 0] == x0_param)
        if bool(cfg.get("enforce_state_bounds", True)):
            opti.subject_to(opti.bounded(system.xmin, X, system.xmax))
        for k in range(horizon):
            opti.subject_to(X[:, k + 1] == f_casadi(X[:, k], U[:, k]))
            opti.subject_to(opti.bounded(system.umin, U[:, k], system.umax))
            if bool(cfg.get("enforce_safety_constraints", True)):
                for expr, delta in h_casadi(X[:, k + 1]):
                    opti.subject_to(expr >= delta)
        opti.minimize(system.casadi_dpc_objective(ca, X, U, ref_param, dict(cfg)))
        opti.solver("ipopt", {"print_time": False}, _ipopt_options(cfg))
        setup_time += _time.perf_counter() - started
        cache[horizon] = {"opti": opti, "X": X, "U": U, "x0": x0_param, "ref": ref_param,
                          "X_last": None, "U_last": None, "lam_last": None}
        return cache[horizon]

    x_history = np.empty((T, nx), dtype=float)
    u_history = np.empty((T - 1, nu), dtype=float)
    horizon_history = np.empty(T - 1, dtype=int)
    x_history[0] = x_initial
    x_current = x_initial.copy()
    solve_times, solver_iterations, solver_statuses = [], [], []
    for step in range(T - 1):
        horizon = _next_reference_change_horizon(reference, step, maximum)
        horizon_history[step] = horizon
        problem = cache.get(horizon) or build_problem(horizon)
        opti, X, U = problem["opti"], problem["X"], problem["U"]
        current_reference = reference[step]
        opti.set_value(problem["x0"], x_current)
        opti.set_value(problem["ref"], current_reference)
        if problem["U_last"] is None:
            if make_u_guess is not None:
                U0 = np.asarray(make_u_guess(x_current, _held_reference(reference, step, horizon)), dtype=float)
            else:
                midpoint = 0.5 * (np.asarray(system.umin) + np.asarray(system.umax))
                U0 = np.broadcast_to(midpoint, (horizon, nu)).copy()
            X0 = np.empty((horizon + 1, nx), dtype=float)
            X0[0] = x_current
            for k in range(horizon):
                X0[k + 1] = np.asarray(f_casadi(X0[k], U0[k])).reshape(nx)
        else:
            U0 = np.concatenate([problem["U_last"][1:], problem["U_last"][-1:]], axis=0)
            X0 = np.concatenate([problem["X_last"][1:], problem["X_last"][-1:]], axis=0)
            X0[0] = x_current
        opti.set_initial(X, X0.T)
        opti.set_initial(U, U0.T)
        if problem["lam_last"] is not None and bool(cfg.get("warm_start", False)):
            opti.set_initial(opti.lam_g, problem["lam_last"])
        started = _time.perf_counter()
        solution = opti.solve()
        solve_times.append(_time.perf_counter() - started)
        stats = solution.stats()
        solver_statuses.append(str(stats.get("return_status", "unknown")))
        solver_iterations.append(int(stats.get("iter_count", -1)))
        problem["X_last"] = np.asarray(solution.value(X), dtype=float).reshape(nx, horizon + 1).T
        problem["U_last"] = np.asarray(solution.value(U), dtype=float).reshape(nu, horizon).T
        if bool(cfg.get("warm_start", False)):
            problem["lam_last"] = np.asarray(solution.value(opti.lam_g), dtype=float)
        u_applied = problem["U_last"][0]
        x_current = np.asarray(f_casadi(x_current, u_applied), dtype=float).reshape(nx)
        u_history[step], x_history[step + 1] = u_applied, x_current

    total_solve_time = float(sum(solve_times))
    statuses = sorted(set(solver_statuses))
    loss_spec = system.dpc_loss_spec(cfg)
    loss_spec["horizon"] = maximum
    return {
        "x_traj": torch.tensor(x_history, dtype=torch.float32).unsqueeze(0),
        "u_traj": torch.tensor(u_history, dtype=torch.float32).unsqueeze(0), "r_traj": data["r"],
        "solve_time_s": total_solve_time, "setup_time_s": setup_time,
        "per_step_s": total_solve_time / max(T - 1, 1),
        "solver_status": statuses[0] if len(statuses) == 1 else ",".join(statuses),
        "solver_iterations": int(sum(max(v, 0) for v in solver_iterations)),
        "solver_iterations_mean": float(np.mean(solver_iterations)) if solver_iterations else 0.0,
        "solver_iterations_max": int(max(solver_iterations, default=0)),
        "solve_times_s": np.asarray(solve_times, dtype=float),
        "horizon_history": horizon_history, "mpc_horizon": maximum,
        "mpc_mode": "receding_horizon_next_reference_change", "dpc_loss_spec": loss_spec,
    }


def _solve_receding_horizon_mpc(system, data, cfg: Dict, hooks: Dict) -> Dict:
    """Online MPC with a fixed DPC-style horizon and current-reference information."""
    import time as _time

    f_casadi = hooks["f_casadi"]
    h_casadi = hooks.get("h_casadi", lambda x: [])
    x_initial = data["xn"][0, 0, :].detach().cpu().numpy().astype(float)
    reference = data["r"][0].detach().cpu().numpy().astype(float)
    T = reference.shape[0]
    nx, nu = system.nx, system.nu
    horizon = int(cfg.get("horizon", cfg.get("nsteps", 50)))
    if horizon < 1:
        raise ValueError("MPC horizon must be positive")
    arrival_deadline = bool(cfg.get("arrival_deadline", False))
    arrival_tolerance = float(cfg.get("arrival_tolerance", 1.0e-2))
    if arrival_tolerance < 0.0:
        raise ValueError("MPC arrival_tolerance must be nonnegative")

    setup_t0 = _time.perf_counter()
    opti = ca.Opti()
    X = opti.variable(nx, horizon + 1)
    U = opti.variable(nu, horizon)
    x0_param = opti.parameter(nx)
    ref_param = opti.parameter(nx)
    # The ordinary DPC objective penalizes only differences *inside* an MPC
    # prediction window.  In receding-horizon use that leaves its first move
    # unconnected to the action applied at the preceding real-time step.  When
    # Q_du is non-zero, include that boundary increment as well.
    q_du = float(cfg.get("weights", {}).get("Q_du", 0.0))
    u_previous_param = opti.parameter(nu) if q_du > 0.0 else None
    terminal_selector = opti.parameter(horizon + 1) if arrival_deadline else None
    opti.subject_to(X[:, 0] == x0_param)
    if bool(cfg.get("enforce_state_bounds", True)):
        opti.subject_to(opti.bounded(system.xmin, X, system.xmax))

    for k in range(horizon):
        opti.subject_to(X[:, k + 1] == f_casadi(X[:, k], U[:, k]))
        opti.subject_to(opti.bounded(system.umin, U[:, k], system.umax))
        if bool(cfg.get("enforce_safety_constraints", True)):
            for expr, delta in h_casadi(X[:, k + 1]):
                opti.subject_to(expr >= delta)
    objective_cfg = dict(cfg)
    if terminal_selector is not None:
        objective_cfg["_terminal_state"] = ca.mtimes(X, terminal_selector)
    obj = system.casadi_dpc_objective(ca, X, U, ref_param, objective_cfg)
    if u_previous_param is not None:
        obj += q_du * ca.sumsqr(U[:, 0] - u_previous_param)
    opti.minimize(obj)
    opti.solver("ipopt", {"print_time": False}, _ipopt_options(cfg))
    setup_time = _time.perf_counter() - setup_t0

    x_history = np.empty((T, nx), dtype=float)
    u_history = np.empty((T - 1, nu), dtype=float)
    x_history[0] = x_initial
    x_current = x_initial.copy()
    previous_X = None
    previous_U = None
    previous_lam_g = None
    solve_times = []
    solver_iterations = []
    solver_statuses = []
    make_u_guess = hooks.get("mpc_u_guess")
    deadline_remaining = horizon
    previous_reference = None

    for step in range(T - 1):
        reference_horizon = _held_reference(reference, step, horizon)
        current_reference = reference_horizon[0]
        reference_changed = (
            previous_reference is None
            or not np.allclose(current_reference, previous_reference, rtol=0.0, atol=1.0e-9)
        )
        if reference_changed:
            deadline_remaining = horizon
        opti.set_value(x0_param, x_current)
        opti.set_value(ref_param, current_reference)
        if terminal_selector is not None:
            error_norm = float(np.linalg.norm(x_current - current_reference))
            terminal_index = 0 if error_norm <= arrival_tolerance else max(deadline_remaining, 1)
            selector = np.zeros(horizon + 1, dtype=float)
            selector[terminal_index] = 1.0
            opti.set_value(terminal_selector, selector)

        if previous_U is None:
            if make_u_guess is not None:
                U0 = np.asarray(make_u_guess(x_current, reference_horizon), dtype=float)
            else:
                midpoint = 0.5 * (np.asarray(system.umin) + np.asarray(system.umax))
                U0 = np.broadcast_to(midpoint, (horizon, nu)).copy()
            X0 = np.empty((horizon + 1, nx), dtype=float)
            X0[0] = x_current
            for k in range(horizon):
                X0[k + 1] = np.asarray(f_casadi(X0[k], U0[k])).reshape(nx)
        else:
            U0 = np.concatenate([previous_U[1:], previous_U[-1:]], axis=0)
            X0 = np.concatenate([previous_X[1:], previous_X[-1:]], axis=0)
            X0[0] = x_current

        opti.set_initial(X, X0.T)
        opti.set_initial(U, U0.T)
        if u_previous_param is not None:
            # No physical action precedes the first rollout sample.  Anchor
            # that first NLP to its feasible initial guess, then anchor every
            # subsequent solve to the action actually executed.
            anchor = U0[0] if previous_U is None else u_history[step - 1]
            opti.set_value(u_previous_param, anchor)
        if previous_lam_g is not None and bool(cfg.get("warm_start", False)):
            opti.set_initial(opti.lam_g, previous_lam_g)

        solve_t0 = _time.perf_counter()
        sol = opti.solve()
        solve_times.append(_time.perf_counter() - solve_t0)
        stats = sol.stats()
        solver_statuses.append(str(stats.get("return_status", "unknown")))
        solver_iterations.append(int(stats.get("iter_count", -1)))

        previous_X = np.asarray(sol.value(X), dtype=float).reshape(nx, horizon + 1).T
        previous_U = np.asarray(sol.value(U), dtype=float).reshape(nu, horizon).T
        if bool(cfg.get("warm_start", False)):
            previous_lam_g = np.asarray(sol.value(opti.lam_g), dtype=float)
        u_applied = previous_U[0]
        x_current = np.asarray(f_casadi(x_current, u_applied), dtype=float).reshape(nx)
        u_history[step] = u_applied
        x_history[step + 1] = x_current
        if terminal_selector is not None and terminal_index > 0:
            deadline_remaining = max(deadline_remaining - 1, 1)
        previous_reference = current_reference.copy()

    total_solve_time = float(sum(solve_times))
    unique_statuses = sorted(set(solver_statuses))
    loss_spec = system.dpc_loss_spec(cfg)
    loss_spec["horizon"] = horizon
    return {
        "x_traj": torch.tensor(x_history, dtype=torch.float32).unsqueeze(0),
        "u_traj": torch.tensor(u_history, dtype=torch.float32).unsqueeze(0),
        "r_traj": data["r"],
        "solve_time_s": total_solve_time,
        "setup_time_s": setup_time,
        "per_step_s": total_solve_time / max(T - 1, 1),
        "solver_status": unique_statuses[0] if len(unique_statuses) == 1 else ",".join(unique_statuses),
        "solver_iterations": int(sum(max(value, 0) for value in solver_iterations)),
        "solver_iterations_mean": float(np.mean(solver_iterations)) if solver_iterations else 0.0,
        "solver_iterations_max": int(max(solver_iterations, default=0)),
        "solve_times_s": np.asarray(solve_times, dtype=float),
        "mpc_horizon": horizon,
        "mpc_mode": "receding_horizon",
        "arrival_deadline": arrival_deadline,
        "arrival_tolerance": arrival_tolerance,
        "dpc_loss_spec": loss_spec,
    }


def _solve_dpc_open_loop_mpc(
    system,
    data,
    cfg: Dict,
    hooks: Dict,
    initial_guess: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    """Optimize and execute complete DPC-horizon action blocks.

    Every nonlinear program has exactly the horizon used during DPC training. The plan is
    executed open loop until all horizon actions are consumed, the real reference changes,
    or the evaluation trajectory ends. A reference change starts a new full-horizon solve;
    future references are never exposed to the optimizer early.
    """
    import time as _time

    f_casadi = hooks["f_casadi"]
    h_casadi = hooks.get("h_casadi", lambda x: [])
    x_initial = data["xn"][0, 0, :].detach().cpu().numpy().astype(float)
    reference = data["r"][0].detach().cpu().numpy().astype(float)
    T = reference.shape[0]
    nx, nu = system.nx, system.nu
    horizon = int(cfg.get("horizon", cfg.get("nsteps", 50)))
    if horizon < 1:
        raise ValueError("MPC horizon must be positive")
    if T < 2:
        raise ValueError("MPC evaluation requires at least one control step")
    constraint_margin = float(cfg.get("constraint_margin", 0.0))
    state_bound_margin = float(cfg.get("state_bound_margin", 0.0))
    if constraint_margin < 0.0 or state_bound_margin < 0.0:
        raise ValueError("MPC constraint margins must be nonnegative")
    enforce_rate = bool(cfg.get("enforce_control_rate_constraints", False))
    enforce_terminal_reference = bool(cfg.get("enforce_terminal_reference", False))
    terminal_reference_tolerance = float(cfg.get("terminal_reference_tolerance", 0.0))
    if terminal_reference_tolerance < 0.0:
        raise ValueError("terminal_reference_tolerance must be nonnegative")
    du_max = cfg.get("du_max")
    if enforce_rate and du_max is None:
        raise ValueError("enforce_control_rate_constraints requires du_max")
    du_max = None if du_max is None else float(du_max)
    rate_margin = float(cfg.get("control_rate_margin", constraint_margin))
    if enforce_rate and not 0.0 <= rate_margin < du_max ** 2:
        raise ValueError("control_rate_margin must satisfy 0 <= margin < du_max**2")

    setup_t0 = _time.perf_counter()
    opti = ca.Opti()
    X = opti.variable(nx, horizon + 1)
    U = opti.variable(nu, horizon)
    x0_param = opti.parameter(nx)
    ref_param = opti.parameter(nx)
    u_previous_param = opti.parameter(nu) if enforce_rate else None
    first_rate_bound_sq = opti.parameter() if enforce_rate else None
    opti.subject_to(X[:, 0] == x0_param)
    if bool(cfg.get("enforce_state_bounds", True)):
        lower_x = float(system.xmin) + state_bound_margin
        upper_x = float(system.xmax) - state_bound_margin
        if lower_x >= upper_x:
            raise ValueError("state_bound_margin leaves no feasible state interval")
        opti.subject_to(opti.bounded(lower_x, X, upper_x))

    for k in range(horizon):
        opti.subject_to(X[:, k + 1] == f_casadi(X[:, k], U[:, k]))
        opti.subject_to(opti.bounded(system.umin, U[:, k], system.umax))
        if bool(cfg.get("enforce_safety_constraints", True)):
            for expr, delta in h_casadi(X[:, k + 1]):
                opti.subject_to(expr >= float(delta) + constraint_margin)
        if enforce_rate:
            previous = u_previous_param if k == 0 else U[:, k - 1]
            bound_sq = first_rate_bound_sq if k == 0 else du_max ** 2 - rate_margin
            opti.subject_to(ca.sumsqr(U[:, k] - previous) <= bound_sq)
    if enforce_terminal_reference:
        if terminal_reference_tolerance == 0.0:
            opti.subject_to(X[:, -1] == ref_param)
        else:
            opti.subject_to(X[:, -1] >= ref_param - terminal_reference_tolerance)
            opti.subject_to(X[:, -1] <= ref_param + terminal_reference_tolerance)
    opti.minimize(system.casadi_dpc_objective(ca, X, U, ref_param, cfg))
    opti.solver("ipopt", {"print_time": False}, _ipopt_options(cfg))
    setup_time = _time.perf_counter() - setup_t0

    x_history = np.empty((T, nx), dtype=float)
    u_history = np.empty((T - 1, nu), dtype=float)
    x_history[0] = x_initial
    x_current = x_initial.copy()
    solve_times = []
    solver_iterations = []
    solver_statuses = []
    block_starts = []
    applied_block_lengths = []
    previous_U = None
    previous_reference = None
    previous_applied = horizon
    previous_applied_u = None
    make_u_guess = hooks.get("mpc_u_guess")
    guess = initial_guess or {}
    first_U = _numpy_guess(guess.get("u_traj"), (horizon, nu), "initial u_traj")
    first_X = _numpy_guess(guess.get("x_traj"), (horizon + 1, nx), "initial x_traj")

    step = 0
    while step < T - 1:
        current_reference = reference[step]
        max_apply = min(horizon, T - 1 - step)
        future = reference[step + 1:step + max_apply + 1]
        changed = np.any(~np.isclose(future, current_reference, rtol=0.0, atol=1.0e-9), axis=1)
        n_apply = int(np.flatnonzero(changed)[0] + 1) if np.any(changed) else max_apply

        opti.set_value(x0_param, x_current)
        opti.set_value(ref_param, current_reference)
        if enforce_rate:
            if previous_applied_u is None:
                # There is no applied action before the first evaluation sample, so the
                # trajectory metric has no initial increment to constrain.
                first_bound = nu * float(np.max(np.asarray(system.umax) - np.asarray(system.umin))) ** 2
                opti.set_value(u_previous_param, np.zeros(nu, dtype=float))
                opti.set_value(first_rate_bound_sq, first_bound)
            else:
                opti.set_value(u_previous_param, previous_applied_u)
                opti.set_value(first_rate_bound_sq, du_max ** 2 - rate_margin)
        same_reference = (
            previous_reference is not None
            and np.allclose(current_reference, previous_reference, rtol=0.0, atol=1.0e-9)
        )
        if step == 0 and first_U is not None:
            U0 = first_U
        elif same_reference and previous_U is not None and previous_applied < horizon:
            tail = previous_U[previous_applied:]
            U0 = np.concatenate(
                [tail, np.repeat(previous_U[-1:], previous_applied, axis=0)], axis=0
            )
        elif make_u_guess is not None:
            held_reference = np.repeat(current_reference.reshape(1, -1), horizon + 1, axis=0)
            U0 = np.asarray(make_u_guess(x_current, held_reference), dtype=float)
        else:
            midpoint = 0.5 * (np.asarray(system.umin) + np.asarray(system.umax))
            U0 = np.broadcast_to(midpoint, (horizon, nu)).copy()

        if U0.shape != (horizon, nu):
            raise ValueError(f"MPC control guess must have shape {(horizon, nu)}, got {U0.shape}")
        if step == 0 and first_X is not None:
            X0 = first_X
        else:
            X0 = np.empty((horizon + 1, nx), dtype=float)
            X0[0] = x_current
            for k in range(horizon):
                X0[k + 1] = np.asarray(f_casadi(X0[k], U0[k])).reshape(nx)
        opti.set_initial(X, X0.T)
        opti.set_initial(U, U0.T)

        solve_t0 = _time.perf_counter()
        sol = opti.solve()
        solve_times.append(_time.perf_counter() - solve_t0)
        stats = sol.stats()
        solver_statuses.append(str(stats.get("return_status", "unknown")))
        solver_iterations.append(int(stats.get("iter_count", -1)))
        X_plan = np.asarray(sol.value(X), dtype=float).reshape(nx, horizon + 1).T
        U_plan = np.asarray(sol.value(U), dtype=float).reshape(nu, horizon).T

        block_starts.append(step)
        applied_block_lengths.append(n_apply)
        for k in range(n_apply):
            u_applied = U_plan[k]
            x_current = np.asarray(f_casadi(x_current, u_applied), dtype=float).reshape(nx)
            u_history[step + k] = u_applied
            x_history[step + k + 1] = x_current
            previous_applied_u = u_applied.copy()

        previous_U = U_plan
        previous_reference = current_reference.copy()
        previous_applied = n_apply
        step += n_apply

    total_solve_time = float(sum(solve_times))
    unique_statuses = sorted(set(solver_statuses))
    loss_spec = system.dpc_loss_spec(cfg)
    loss_spec["horizon"] = horizon
    return {
        "x_traj": torch.tensor(x_history, dtype=torch.float32).unsqueeze(0),
        "u_traj": torch.tensor(u_history, dtype=torch.float32).unsqueeze(0),
        "r_traj": data["r"],
        "solve_time_s": total_solve_time,
        "setup_time_s": setup_time,
        "per_step_s": total_solve_time / max(T - 1, 1),
        "solver_status": unique_statuses[0] if len(unique_statuses) == 1 else ",".join(unique_statuses),
        "solver_iterations": int(sum(max(value, 0) for value in solver_iterations)),
        "solver_iterations_mean": float(np.mean(solver_iterations)) if solver_iterations else 0.0,
        "solver_iterations_max": int(max(solver_iterations, default=0)),
        "solve_times_s": np.asarray(solve_times, dtype=float),
        "mpc_horizon": horizon,
        "mpc_mode": "dpc_open_loop",
        "num_solves": len(solve_times),
        "block_starts": np.asarray(block_starts, dtype=int),
        "applied_block_lengths": np.asarray(applied_block_lengths, dtype=int),
        "constraint_margin": constraint_margin,
        "state_bound_margin": state_bound_margin,
        "control_rate_margin": rate_margin if enforce_rate else 0.0,
        "enforce_terminal_reference": enforce_terminal_reference,
        "terminal_reference_tolerance": terminal_reference_tolerance,
        "dpc_loss_spec": loss_spec,
    }


def solve_mpc(
    system,
    data: Dict[str, torch.Tensor],
    cfg: Dict,
    *,
    initial_guess: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    """Solve the configured tracking MPC from ``data['xn']`` over ``data['r']``.

    ``mode='dpc_open_loop'`` optimizes the complete DPC training horizon and executes its
    actions open loop. It solves again only after that block or when the real reference
    changes. ``mode='receding_horizon'`` remains available for explicit online MPC runs.
    """
    if not _HAS_CASADI:
        raise RuntimeError("solve_mpc requires CasADi (`pip install casadi`).")
    hooks = system.casadi_hooks(
        cfg, integration_method=cfg.get("integration_method", "rk4")
    )
    if hooks is None or "f_casadi" not in hooks:
        raise NotImplementedError(f"system {system.name!r} provides no CasADi dynamics hook")

    mode = str(cfg.get("mode", "full_horizon")).lower()
    if mode in {"dpc_open_loop", "open_loop_blocks", "dpc_horizon"}:
        return _solve_dpc_open_loop_mpc(system, data, cfg, hooks, initial_guess)
    if mode in {"receding", "receding_horizon", "online"}:
        schedule = str(cfg.get("horizon_schedule", "fixed")).lower()
        if schedule in {"next_reference_change", "reference_change", "adaptive_reference"}:
            return _solve_reference_change_horizon_mpc(system, data, cfg, hooks)
        if schedule not in {"fixed", "constant"}:
            raise ValueError(
                "horizon_schedule must be 'fixed' or 'next_reference_change'"
            )
        return _solve_receding_horizon_mpc(system, data, cfg, hooks)
    if mode not in {"full", "full_horizon", "open_loop"}:
        raise ValueError(
            "MPC mode must be 'dpc_open_loop', 'full_horizon', or 'receding_horizon'"
        )

    f_casadi = hooks["f_casadi"]
    h_casadi = hooks.get("h_casadi", lambda x: [])

    x0 = data["xn"][0, 0, :].detach().cpu().numpy()
    R = data["r"][0].detach().cpu().numpy()
    T = R.shape[0]
    nx, nu = system.nx, system.nu
    w = cfg.get("weights", {})
    Qr = float(w.get("Q_r", 1.0))
    Qu = float(w.get("Q_u", 0.01))
    Qdu = float(w.get("Q_du", 0.0))

    opti = ca.Opti()
    X = opti.variable(nx, T)
    U = opti.variable(nu, T - 1)
    opti.subject_to(X[:, 0] == x0)
    if bool(cfg.get("enforce_state_bounds", True)):
        opti.subject_to(opti.bounded(system.xmin, X, system.xmax))

    obj = 0
    for k in range(T - 1):
        opti.subject_to(X[:, k + 1] == f_casadi(X[:, k], U[:, k]))
        opti.subject_to(opti.bounded(system.umin, U[:, k], system.umax))
        for expr, delta in h_casadi(X[:, k + 1]):
            opti.subject_to(expr >= delta)
        obj += Qu * ca.sumsqr(U[:, k])
        if k > 0 and Qdu > 0.0:
            obj += Qdu * ca.sumsqr(U[:, k] - U[:, k - 1])
    for k in range(T):
        rk = R[k] if R.shape[1] == nx else np.concatenate([R[k], np.zeros(nx - R.shape[1])])
        obj += Qr * ca.sumsqr(X[:, k] - rk)
    opti.minimize(obj)

    opti.solver(
        "ipopt",
        {"print_time": False},
        _ipopt_options(cfg),
    )

    guess = initial_guess or {}
    X0 = _numpy_guess(guess.get("x_traj"), (T, nx), "initial x_traj")
    U0 = _numpy_guess(guess.get("u_traj"), (T - 1, nu), "initial u_traj")
    if U0 is None:
        make_u_guess = hooks.get("mpc_u_guess")
        if make_u_guess is not None:
            U0 = np.asarray(make_u_guess(x0, R), dtype=float)
        else:
            u_mid = 0.5 * (np.asarray(system.umin) + np.asarray(system.umax))
            U0 = np.broadcast_to(u_mid, (T - 1, nu)).copy()
    if X0 is None:
        X0 = np.empty((T, nx), dtype=float)
        X0[0] = x0
        for k in range(T - 1):
            X0[k + 1] = np.asarray(f_casadi(X0[k], U0[k])).reshape(nx)
    opti.set_initial(X, X0.T)
    opti.set_initial(U, U0.T)

    import time as _time
    t0 = _time.perf_counter()
    sol = opti.solve()
    elapsed = _time.perf_counter() - t0
    stats = sol.stats()

    Xsol = np.array(sol.value(X)).reshape(nx, T).T
    Usol = np.array(sol.value(U)).reshape(nu, T - 1).T
    return {
        "x_traj": torch.tensor(Xsol, dtype=torch.float32).unsqueeze(0),
        "u_traj": torch.tensor(Usol, dtype=torch.float32).unsqueeze(0),
        "r_traj": data["r"],
        "solve_time_s": elapsed,
        "per_step_s": elapsed / max(T - 1, 1),
        "solver_status": stats.get("return_status", "unknown"),
        "solver_iterations": int(stats.get("iter_count", -1)),
    }
