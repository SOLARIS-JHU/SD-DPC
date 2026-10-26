"""Evaluation: metrics, closed-loop rollouts, and the comparison orchestrator."""
from .metrics import (
    compute_metrics,
    control_smoothness,
    policy_forward_latency,
    policy_memory_footprint,
    task_loss,
)
from .rollout import rollout_closed_loop, make_test_data
from .evaluate import evaluate, default_methods, dpc_loss_reference, sample_scenario, aggregate
from .twotank import (
    make_twotank_scenarios,
    evaluate_twotank_nominal,
    violation_summary,
)

__all__ = [
    "compute_metrics",
    "control_smoothness",
    "policy_forward_latency",
    "policy_memory_footprint",
    "task_loss",
    "rollout_closed_loop",
    "make_test_data",
    "evaluate",
    "default_methods",
    "dpc_loss_reference",
    "sample_scenario",
    "aggregate",
    "make_twotank_scenarios",
    "evaluate_twotank_nominal",
    "violation_summary",
]
