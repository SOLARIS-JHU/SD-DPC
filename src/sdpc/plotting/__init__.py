"""Plotting: trajectories, coefficient plots and reports, and animation."""
from .trajectories import (
    plot_trajectories_with_ellipse,
    plot_training_sample_with_ellipse,
    plot_states_and_controls,
    plot_controller_evaluation,
    plot_trajectories_with_obstacles,
    make_test_sample_discrete,
    make_test_sample,
)
from .report import plot_coef_heatmap, format_metrics_table
from .animation import animate_trajectories_with_ellipse, animate_trajectories_with_obstacles

__all__ = [
    "plot_trajectories_with_ellipse",
    "plot_training_sample_with_ellipse",
    "plot_states_and_controls",
    "plot_controller_evaluation",
    "plot_trajectories_with_obstacles",
    "make_test_sample_discrete",
    "make_test_sample",
    "plot_coef_heatmap",
    "format_metrics_table",
    "animate_trajectories_with_ellipse",
    "animate_trajectories_with_obstacles",
]
