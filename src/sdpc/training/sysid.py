"""System identification: build the ADAM-SINDy problem and train it (Algorithm 1).

The identified model is trained through *rollout* error (one-step + full-horizon prediction)
rather than regression on differentiated data, so it is consistent with how it is later used
for prediction during policy learning (Sec. 3.1). The same one-step map returned by
``System.discrete_step`` is used here and in evaluation, keeping train/deploy consistent.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from neuromancer.system import Node, System
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem

from ..sindy.model import SINDyVectorized
from .trainer import SparseTrainer

__all__ = ["build_sysid_problem", "train_sysid"]


def build_sysid_problem(system, sindy: SINDyVectorized, cfg: Dict, nsteps: int) -> Problem:
    """One-step + full-horizon rollout loss with an L1 penalty on the SINDy coefficients."""
    step_fn = system.discrete_step(sindy)
    node = Node(step_fn, ["xn", "u"], ["xn"], name="x_integrator")
    dyn_sys = System([node], nsteps=nsteps)

    w = cfg["weights"]
    x = variable("X")
    x_hat = variable("xn")[:, :-1, :]

    onestep = w["onestep_coef"] * ((x_hat[:, 1, :] == x[:, 1, :]) ^ 2)
    onestep.name = "onestep_loss"
    reference = w["ref_coef"] * ((x_hat == x) ^ 2)
    reference.name = "reference_loss"
    l1 = w["l1_coef"] * (variable([x], lambda _x: sum(torch.norm(p, p=1) for p in sindy.Xi)) == 0)
    l1.name = "loss_l1_sindy"

    loss = PenaltyLoss([onestep, reference, l1], [])
    return Problem([dyn_sys], loss)


def train_sysid(system, sindy, train_loader, dev_loader, cfg, device, logger=None) -> Dict:
    """Train ``sindy`` on the system-ID data; returns the best model state dict."""
    nsteps = cfg.get("nsteps", 2)
    problem = build_sysid_problem(system, sindy, cfg, nsteps)
    opt = torch.optim.AdamW(sindy.parameters(), lr=cfg.get("lr", 1e-3),
                            weight_decay=cfg.get("weight_decay", 0.01))

    trainer = SparseTrainer(
        problem=problem, sindy=sindy, lr=cfg.get("lr", 1e-3),
        train_data=train_loader, dev_data=dev_loader, optimizers=opt,
        epochs=cfg.get("epochs", 20000), train_metric="train_loss", eval_metric="dev_loss",
        logger=logger, device=device,
        threshold=cfg.get("threshold", 1e-2), prune_every=cfg.get("prune_every", 5000),
        warmup=cfg.get("warmup", 300), patience=cfg.get("patience", 50),
        threshold_mult=cfg.get("threshold_mult", 1.05), prune_noise=cfg.get("prune_noise", 0.05),
        max_unchange=cfg.get("max_unchange", 0),
        l1_decay=cfg.get("l1_decay", 0.3),
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    best = trainer.train()
    problem.load_state_dict(best, strict=False)
    return best
