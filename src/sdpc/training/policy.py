"""Sparse dictionary policy learning via differentiable predictive control (Sec. 3.3).

Generalises the two notebook builders (``SparsePolicyBuilder`` / ``SparsePolicyBuilder2D``)
into one config-driven trainer: the closed-loop graph (policy -> one-step model, plus an
optional ``u_f = pi(r, r)`` node) is assembled here, while the *objectives and constraints*
come from ``System.build_dpc_loss`` — so the DoubleIntegrator (obstacle) and Two-Tank/Van
der Pol (tracking) losses live with their systems and this module stays generic. Works with
both sparse SINDy policies (thresholded training + L1) and NN baselines (plain trainer).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from neuromancer.system import Node, System
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer

from ..actions import ACTION_GRADIENT_MODES, bounded_action
from ..sindy.model import SINDyVectorized
from .trainer import SparseTrainer

__all__ = ["build_policy_problem", "train_policy"]


def build_policy_problem(system, dynamics_model, policy, cfg: Dict, nsteps: int,
                         action_scale: float = 1.0) -> Problem:
    """Assemble the closed-loop DPC :class:`Problem` for a policy (sparse or NN)."""
    umin, umax = system.umin, system.umax
    is_sparse = isinstance(policy, SINDyVectorized)
    action_gradient_mode = str(cfg.get("action_gradient_mode", "straight_through"))
    action_gradient_band = float(cfg.get("action_gradient_band", 0.1))
    action_gradient_leak = float(cfg.get("action_gradient_leak", 0.05))
    if action_gradient_mode not in ACTION_GRADIENT_MODES:
        raise ValueError(
            f"action_gradient_mode must be one of {sorted(ACTION_GRADIENT_MODES)}"
        )

    if is_sparse:
        policy_node = Node(
            lambda xn, r: bounded_action(
                policy(xn, r), umin, umax, action_scale,
                gradient_mode=action_gradient_mode,
                gradient_band=action_gradient_band,
                gradient_leak=action_gradient_leak,
            ),
            ["xn", "r"], ["u"], name="policy_combined",
        )
    else:
        policy_node = policy  # already a Neuromancer Node

    nodes = [policy_node, Node(system.discrete_step(dynamics_model), ["xn", "u"], ["xn"],
                              name="x_integrator")]

    if system.uses_u_at_ref():
        if is_sparse:
            u_at_ref = Node(
                lambda r: bounded_action(
                    policy(r, r), umin, umax, action_scale,
                    gradient_mode=action_gradient_mode,
                    gradient_band=action_gradient_band,
                    gradient_leak=action_gradient_leak,
                ),
                ["r"], ["u_f"], name="policy_at_ref",
            )
        else:
            u_at_ref = Node(
                lambda r: torch.clamp(policy.callable(torch.cat([r, r], dim=-1)), umin, umax),
                ["r"], ["u_f"], name="policy_at_ref",
            )
        nodes.append(u_at_ref)

    cl_system = System(nodes, nsteps=nsteps)
    objectives, constraints = system.build_dpc_loss(cfg)

    if is_sparse:
        l1 = cfg["weights"].get("l1_coef", 0.0) * (
            variable([variable("xn")], lambda _x: sum(torch.norm(p, p=1) for p in policy.Xi)) == 0
        )
        l1.name = "loss_l1_policy"
        objectives = [*objectives, l1]

    loss = PenaltyLoss(objectives, constraints)
    return Problem([cl_system], loss)


def train_policy(system, dynamics_model, policy, train_loader, dev_loader, cfg, device,
                 logger=None, action_scale: float = 1.0):
    """Train the policy in closed loop; returns the best model state dict."""
    nsteps = cfg.get("nsteps", 100)
    problem = build_policy_problem(system, dynamics_model, policy, cfg, nsteps, action_scale)
    lr = cfg.get("lr", 2e-3)

    if isinstance(policy, SINDyVectorized):
        if logger is not None and hasattr(logger, "stdout"):
            default_metrics = [
                "train_loss", "dev_loss",
                "train_task_loss", "dev_task_loss",
                "train_tracking_loss", "dev_tracking_loss",
                "train_l1_loss", "dev_l1_loss",
                "train_constraint_loss", "dev_constraint_loss",
                "train_total_control_action_l1", "dev_total_control_action_l1",
                "dev_x_min_loss", "dev_x_max_loss",
                "dev_y_N_min_loss", "dev_y_N_max_loss",
                "learning_rate", "coefficient_l1_norm",
                "unchanged_prune_count",
                "active_terms_u0", "active_terms_u1", "active_terms_total",
            ]
            requested_metrics = cfg.get("logger_metrics", default_metrics)
            logger.stdout = list(dict.fromkeys([*logger.stdout, *requested_metrics]))
        opt = torch.optim.AdamW(policy.parameters(), lr=lr,
                                weight_decay=cfg.get("weight_decay", 0.01))
        trainer = SparseTrainer(
            problem=problem, sindy=policy, lr=lr, train_data=train_loader, dev_data=dev_loader,
            optimizers=opt, epochs=cfg.get("epochs", 15000),
            train_metric="train_loss", eval_metric="dev_loss", logger=logger, device=device,
            clip=cfg.get("grad_clip", 100.0),
            threshold=cfg.get("threshold", 5e-3), prune_every=cfg.get("prune_every", 1500),
            prune_every_min=cfg.get("prune_every_min", 1200),
            prune_every_decay=cfg.get("prune_every_decay", 100),
            change_prune_every=cfg.get("change_prune_every", True),
            threshold_mult=cfg.get("threshold_mult", 1.5),
            threshold_max=cfg.get("threshold_max", 7e-3),
            max_unchange=cfg.get("max_unchange", 0),
            prune_noise=cfg.get("prune_noise", 0.01),
            lr_decay_gamma=cfg.get("lr_decay_gamma", 0.95),
            lr_decay_step=cfg.get("lr_decay_step", 50),
            lr_scheduler=cfg.get("lr_decay_enabled", False),
            l1_decay=cfg.get("l1_decay", 0.9),
            proximal_l1=(
                cfg["weights"].get("l1_coef", 0.0)
                if cfg.get("proximal_l1", False) else 0.0
            ),
            weight_decay=cfg.get("weight_decay", 0.01),
        )
        best = trainer.train()
        debias = cfg.get("debias", {})
        if debias.get("enabled") and debias.get("require_plateau", True):
            required = int(debias.get("stable_pruning_stages", cfg.get("max_unchange", 0)))
            if trainer.unchanged_prune_count < required:
                raise RuntimeError(
                    f"Sparse phase reached only {trainer.unchanged_prune_count} "
                    f"stable pruning stages; {required} are required for debiasing"
                )
        problem.load_state_dict(best, strict=False)
        return best
    else:
        trainer = Trainer(
            problem, train_data=train_loader, dev_data=dev_loader,
            optimizer=torch.optim.AdamW(policy.parameters(), lr=lr,
                                        weight_decay=cfg.get("weight_decay", 0.01)),
            epochs=cfg.get("epochs", 1000), train_metric="train_loss", eval_metric="dev_loss",
            warmup=cfg.get("warmup", 0), patience=cfg.get("patience", 100), logger=logger,
        )
        best = trainer.train()
        problem.load_state_dict(best, strict=False)
        return best
