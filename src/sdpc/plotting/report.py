"""Coefficient plots and fixed-controller metric tables."""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import torch

__all__ = ["plot_coef_heatmap", "format_metrics_table"]


def _math_term(name: str) -> str:
    """Convert the library's compact term names to Matplotlib math text."""
    if name == "1":
        return r"$1$"
    expression = name.replace("·", r"\,")
    expression = re.sub(r"sqrt\((x\d+)\)", r"\\sqrt{\1}", expression)
    expression = re.sub(
        r"(sin|cos)\((.*?)\)",
        lambda match: rf"\{match.group(1)}\!\left({match.group(2)}\right)",
        expression,
    )
    expression = re.sub(
        r"([xru])([0-9]+)",
        lambda match: rf"{match.group(1)}_{{{int(match.group(2)) + 1}}}",
        expression,
    )
    expression = expression.replace("*", r"\,")
    expression = re.sub(r"\^([0-9]+)", r"^{\1}", expression)
    return f"${expression}$"


def plot_coef_heatmap(
    policy,
    *,
    title: str = "Active policy coefficients",
    annotate: bool = True,
    coefficient_format: str = ".3g",
):
    """Plot the active policy matrix with readable symbolic term labels.

    Zero rows are omitted, the color scale is symmetric about zero, and optional
    cell annotations expose the numerical controller parameters directly.
    """
    import matplotlib.pyplot as plt

    C = policy.coef.detach().cpu().numpy()  # (n_terms, n_out)
    names = policy.function_names
    active = np.where(np.abs(C).sum(axis=1) > 0)[0]
    if active.size == 0:
        raise ValueError("policy has no active coefficients to plot")
    C = C[active]
    labels = [_math_term(names[i]) for i in active]

    limit = float(np.abs(C).max())
    fig_height = min(8.0, max(2.7, 0.42 * len(active) + 1.2))
    fig, ax = plt.subplots(figsize=(4.8, fig_height))
    im = ax.imshow(
        C, aspect="auto", interpolation="nearest", cmap="RdBu_r",
        vmin=-limit, vmax=limit,
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    out_names = policy.policy_name or [f"u{j}" for j in range(policy.n_out)]
    ax.set_xticks(range(policy.n_out))
    ax.set_xticklabels([_math_term(name) for name in out_names], fontsize=10)
    ax.set_xlabel("Policy output")
    ax.set_ylabel("Active dictionary term")
    ax.set_title(title, pad=10)

    if annotate and C.size <= 120:
        threshold = 0.52 * limit
        for row in range(C.shape[0]):
            for col in range(C.shape[1]):
                value = C[row, col]
                ax.text(
                    col, row, format(value, coefficient_format),
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > threshold else "black",
                )

    colorbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.04)
    colorbar.set_label("Coefficient value")
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.45")
    fig.tight_layout(pad=0.8)
    return fig, ax


def format_metrics_table(summary: Dict[str, Dict[str, float]],
                         metrics: List[str] = None) -> str:
    """Render the aggregated metric summary as a plain-text table (mean +/- std)."""
    metrics = metrics or [
        "final_tracking_mse", "steps_to_reach", "total_control_action_l1",
        "control_smoothness_rms", "num_violations", "min_safety_margin",
        "policy_forward_per_step_ms", "sparsity_total_terms",
    ]
    header = ["method"] + metrics
    widths = [max(len(h), 16) for h in header]
    lines = [" | ".join(h.ljust(w) for h, w in zip(header, widths))]
    lines.append("-+-".join("-" * w for w in widths))
    for method, s in summary.items():
        cells = [method.ljust(widths[0])]
        for k, w in zip(metrics, widths[1:]):
            mean, std = s.get(f"{k}_mean"), s.get(f"{k}_std")
            cells.append(("--" if mean is None else f"{mean:.3g}+-{std:.2g}").ljust(w))
        lines.append(" | ".join(cells))
    return "\n".join(lines)
